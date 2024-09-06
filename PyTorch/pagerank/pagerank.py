import json
import os
import time
import torch
from torch_ppr import page_rank
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import pyarrow.fs as fs
import pyarrow.csv as pv
import gc


class GraphEdgeDataset(Dataset):
    def __init__(self, chunk):
        self.edges = self.load_chunk(chunk)

    def load_chunk(self, chunk):
        nodes1 = chunk.column('node1').to_pylist()
        nodes2 = chunk.column('node2').to_pylist()
        chunk_correct_format = [nodes1, nodes2]

        return torch.tensor(data=chunk_correct_format)

    def __len__(self):
        return self.edges.size(1)

    def __getitem__(self, idx):
        return self.edges[:, idx]



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.1'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group - gloo == cpu (not gpu)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    # Make sure all the files used to save the intermediate results are deleted during the cleanup
    directory =  os.path.expanduser('~/PyTorch/pagerank/intermediate_results')
    if os.path.exists(directory):
        # List all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
    
    # Destroy the process group we setup
    dist.destroy_process_group()


# Display and save the results
def display_results(config, world_size, start_time, end_time, scores_dict):
    data_file = config["datafile"].split('.')[0]    # keep only data file name

    results_text = (
        f"\nFor file {data_file} - number of worker machines {world_size} - batch size {config['batch_size']}: \n"
        f"Time taken (PyTorch): {end_time - start_time} seconds\n"
        f"Top 10 scores: {scores_dict} \n"
    )

    print(results_text)
    
    # Create custom file name in results directory, in order to save results for different data sizes and number of machines
    directory = os.path.expanduser('~/PyTorch/pagerank/res')
    file_name = f"{data_file}_{world_size}nodes_results.txt"

    file_path = os.path.join(directory, file_name)
    
    if not os.path.exists(directory):
        # Create a new directory if it does not exist
        os.makedirs(directory)

    
    # Write results to the custom text file - keep the previous results if they exist in the file
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(results_text)
    else:
        with open(file_path, 'w') as f:
            f.write(results_text)


# Create node map and transform data in the right format for torch_ppr pagerank
def input_format(chunk):
    nodes1 = chunk[0].tolist()
    nodes2 = chunk[1].tolist()

    # create set to get rid of duplicates
    all_nodes_set = set(nodes1 + nodes2)
    
    # create node dictionary to know which node corresponds to which index
    all_nodes = list(all_nodes_set)
    all_nodes.sort()
    node_map = {node: idx for idx, node in enumerate(all_nodes)}

    # swap node ids with consecutive indexes starting from 0
    mapped_nodes1 = [node_map[node] for node in nodes1]
    mapped_nodes2 = [node_map[node] for node in nodes2]

    nodes1_tensor = torch.tensor(mapped_nodes1, dtype=torch.int32)
    nodes2_tensor = torch.tensor(mapped_nodes2, dtype=torch.int32)

    return torch.stack([nodes1_tensor, nodes2_tensor], dim=0), all_nodes


# helper function to add nodes and corresponding scores to existing score dictionary
def add_to_dict(global_pr, scores, nodes):
    for idx, node in enumerate(nodes):
        if node in global_pr:
            global_pr[node] += scores[idx]
        else:
            global_pr[node] = scores[idx]
                
    return global_pr


# normalize the scores to all sum to 1
def normalize_pr(scores):
    total_score = scores.sum()

    normalized_scores = scores / total_score

    return normalized_scores


# add dictionary to existing score dictionary
def aggregate_pr_results(global_pr_dict, new_pr_dict):
    for node, score in new_pr_dict.items():
        if node in global_pr_dict:
            global_pr_dict[node] += score
        else:
            global_pr_dict[node] = score
            
    return global_pr_dict

def top_scores(N, scores_dict):
    res = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[:N]
    top_N_scores = dict(res)
    
    return top_N_scores

# save the intermediate dictionary results in files in order to relief memory
def save_intermediate_results(intermediate_result, filename):
    directory = os.path.expanduser('~/PyTorch/pagerank/intermediate_results')
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)

    with open(file_path, 'w') as f:
        json.dump(intermediate_result, f)


# load all the intermediate dictionaries in one dictionary to get the real, aggregated result
def load_intermediate_results(rank, world_size):
    top_aggregated_result = {}
    directory = os.path.expanduser('~/PyTorch/pagerank/intermediate_results')

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                result_dict = json.load(f)
                top_aggregated_result = aggregate_pr_results(top_aggregated_result, result_dict)
                del result_dict
                # keep the 10.000 top scores
                top_aggregated_result = top_scores(10000, top_aggregated_result)
                gc.collect()

    return top_aggregated_result


def distributed_pagerank(rank, world_size):
    # Uncomment only the datafile you want to use
    #datafile = "test_data.csv"  # 10   MB
    #datafile = "data_1.csv"     # 1    GB
    #datafile = "data_2.csv"     # 2.6  GB
    datafile = "data.csv"       # 10.5 GB

    config = {
        "datafile" : datafile,
        "batch_size" : 1024 * 1024 * 50 ,  # 50MB chunks - Adjust as needed
        "hdfs_host" : '192.168.0.1',
        "hdfs_port" : 50000
    }

    # Record start time
    start_time = time.time()

    # Create the distributed system group
    # with the number of machines that is defined from the torchrun command
    setup(rank, world_size)
    
    # Connect to HDFS using PyArrow's FileSystem
    hdfs = fs.HadoopFileSystem(host=config["hdfs_host"], port=config["hdfs_port"])
    file_to_read = f'/data/{config["datafile"]}'

    global_pr = {}
    with hdfs.open_input_file(file_to_read) as file:
        # Define CSV read options to read in chunks
        read_options = pv.ReadOptions(block_size=config["batch_size"])  # 50 MB chunks
        csv_reader = pv.open_csv(file, read_options=read_options)

        for i, chunk in enumerate(csv_reader):
            print("\n\nChunk", i)
            # create the dataset of the 50MB chunks - sampler - dataloader
            dataset = GraphEdgeDataset(chunk)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle = False)
            dataloader = DataLoader(dataset, batch_size=1024 * 1024, sampler=sampler, shuffle = False)

            for batch in dataloader:
                print("Batch")
                batch = batch.t()
                pr_input, nodes = input_format(batch)
                pagerank_results = page_rank(edge_index = pr_input)
                scores = pagerank_results.tolist()
                global_pr = add_to_dict(global_pr, scores, nodes)

            # every 10 chunks save result dictionary to relief memory
            if i % 10 == 0:
                save_intermediate_results(global_pr, f"result_chunk_{i}.json")
                # Clear current results to free memory
                global_pr = {}
                gc.collect()
    
    if global_pr != {} :
        save_intermediate_results(global_pr, f"result_chunk_last.json")

    gc.collect()
    
    final_aggregated_result = load_intermediate_results(rank, world_size)

    nodes = []
    scores = []
    for key, value in final_aggregated_result.items():
        nodes.append(int(key))
        scores.append(value)

    gc.collect()
    
    # Convert lists to tensors
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    nodes_tensor = torch.tensor(nodes, dtype=torch.int32)

    tensor_local_pr = torch.stack([scores_tensor, nodes_tensor], dim=0)

    # Get the size of the first dimension (number of nodes) for local tensor
    local_size = torch.tensor([tensor_local_pr.size(1)], dtype=torch.long)
    size_list = [torch.tensor([0], dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    # Find the maximum size
    max_size = max([size.item() for size in size_list])

    # Pad local tensors to max size (for scores and nodes separately)
    if local_size.item() < max_size:
        padding_size = max_size - local_size.item()
        # Pad scores with zeros, keeping type float32
        padded_scores = torch.cat([tensor_local_pr[0], torch.zeros(padding_size, dtype=torch.float32)], dim=0)
        # Pad nodes with a dummy value (-1), keeping type int
        padded_nodes = torch.cat([tensor_local_pr[1], torch.full((padding_size,), -1, dtype=torch.int32)], dim=0)

        tensor_local_pr = torch.stack([padded_scores, padded_nodes], dim=0)

    # Prepare list to gather all tensors
    gather_list = [torch.stack( [torch.zeros(max_size, dtype=torch.float32), torch.zeros(max_size, dtype=torch.int32)] ) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_local_pr)

    # handle gathered results in the master node only
    if rank == 0:
        # Separate scores and nodes
        all_scores = torch.cat([t[0] for t in gather_list], dim=0)
        all_nodes = torch.cat([t[1] for t in gather_list], dim=0)

        # Filter out padding values
        valid_nodes_mask = all_nodes != -1
        valid_scores_mask = all_scores != 0.0

        filtered_nodes = all_nodes[valid_nodes_mask]
        filtered_scores = all_scores[valid_scores_mask]

        # gather results in one dictionary - add scores for same node
        global_pr = {}
        global_pr = add_to_dict(global_pr, filtered_scores.tolist(), filtered_nodes.tolist())
        top_global_pr = top_scores(10000, global_pr)
        
        # convert to lists and then tensors to handle results easier
        nodes = []
        scores = []          
        for key, value in top_global_pr.items():
            nodes.append(int(key))
            scores.append(value)
            
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        # normalize the gathered final results 
        pr_scores = normalize_pr(scores_tensor)
        
        # get the 10 most significant nodes to display in the results
        res = pr_scores.argsort(dim = 0, descending=True)[:10]
        result_scores = {}
        for i in res:
            result_scores[nodes[i]] = pr_scores[i].item()
    
    # Record end time
    end_time = time.time()

    # Display the results after the time recording has ended in master node only
    if rank == 0:
        display_results(config, world_size, start_time, end_time, result_scores)

    # Cleanup after the finishing of the program
    cleanup()


def main():
    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))

    distributed_pagerank(rank, world_size)

if __name__ == "__main__":
    main()
