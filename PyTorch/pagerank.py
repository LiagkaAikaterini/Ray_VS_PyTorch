import os
import time
import torch
from torch_ppr import page_rank
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import pyarrow.fs as fs
import pyarrow.csv as pv

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

    # initialize the process group - gloo == cpu not gpu
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
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
    directory = os.path.expanduser('~/pytorch/pagerank/res')
    file_name = f"{data_file}_{world_size}nodes_results.txt"

    file_path = os.path.join(directory, file_name)
    
    if not os.path.exists(directory):
        # Create a new directory because it does not exist
        os.makedirs(directory)

    
    # Write results to the custom text file
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(results_text)
    else:
        with open(file_path, 'w') as f:
            f.write(results_text)

def get_node_of_interest(chunk):
    
    nodes1 = chunk[0].tolist()
    nodes2 = chunk[1].tolist()
    
    # create set to get rid of duplicates
    all_nodes_set = set(nodes1 + nodes2)
    # create node dictionary to know which node corresponds to which 
    all_nodes = list(all_nodes_set)
    all_nodes.sort()
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    
    mapped_nodes1 = [node_map[node] for node in nodes1]
    mapped_nodes2 = [node_map[node] for node in nodes2]
    
    nodes1_tensor = torch.tensor(mapped_nodes1, dtype=torch.int32)
    nodes2_tensor = torch.tensor(mapped_nodes2, dtype=torch.int32)

    return torch.stack([nodes1_tensor, nodes2_tensor], dim=0), all_nodes


def result_format(nodes_list, pagerank_scores):
    scores = pagerank_scores.tolist()
    
    scores_tensor = torch.tensor(scores, dtype=torch.float32)  
    nodes_tensor = torch.tensor(nodes_list, dtype=torch.int32)

    correct_format = torch.stack([scores_tensor, nodes_tensor])
        
    return correct_format
       

def aggregate_ppr_results(local_ppr_list):
    global_ppr = {}
    for local_ppr_tensor in local_ppr_list:
        
        local_ppr = local_ppr_tensor.tolist()
        nodes = local_ppr[1]
        scores = local_ppr[0]
        
        for idx, node in enumerate(nodes):
            if node in global_ppr:
                global_ppr[node] += scores[idx]
            else:
                global_ppr[node] = scores[idx]
    
    nodes = []
    scores = []          
    for key, value in global_ppr.items():
        nodes.append(key)
        scores.append(value)
        
    # Convert lists to tensors
    scores_tensor = torch.tensor(scores, dtype=torch.float32)  # Assuming scores are floats
    nodes_tensor = torch.tensor(nodes, dtype=torch.int32)       # Assuming nodes are integers or can be encoded as such

    return torch.stack([scores_tensor, nodes_tensor], dim=0)

def normalize_ppr(global_ppr):
    scores = global_ppr[0]
    nodes = global_ppr[1]

    total_score = scores.sum()

    normalized_scores = scores / total_score

    return normalized_scores, nodes

 
def distributed_pagerank(rank, world_size):
    # Uncomment only the datafile you want to use
    #datafile = "test_data.csv"  # 10   MB
    datafile = "data_1.csv"     # 1    GB
    #datafile = "data_2.csv"     # 2.6  GB
    #datafile = "data.csv"       # 10.5 GB
    
    config = {
        "datafile" : datafile,
        "batch_size" : 1024 * 1024 * 50 ,  # 1MB chunks - Adjust as needed
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

    with hdfs.open_input_file(file_to_read) as file:
        # Define CSV read options to read in chunks
        read_options = pv.ReadOptions(block_size=config["batch_size"])  # 50 MB chunks
        csv_reader = pv.open_csv(file, read_options=read_options)
        
        #results = {}
        results = []
        for chunk in csv_reader:
            print("\n\nChunk")
            # create the dataset of the 50MB chunks - sampler - dataloader 
            dataset = GraphEdgeDataset(chunk)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle = False)
            dataloader = DataLoader(dataset, batch_size=1024 * 1024, sampler=sampler, shuffle = False)
            
            for batch in dataloader:
                print("Batch")
                batch = batch.t()
                input1, all_nodes = get_node_of_interest(batch)
                pagerank_results = page_rank(edge_index = input1)
                result = result_format(all_nodes, pagerank_results)
                results.append(result)
    #------------------------------------------------------------------------------------
    tensor_local_ppr = aggregate_ppr_results(results)

    # Get the size of the first dimension (number of nodes) for local tensor
    local_size = torch.tensor([tensor_local_ppr.size(1)], dtype=torch.long)
    size_list = [torch.tensor([0], dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    # Find the maximum size
    max_size = max([size.item() for size in size_list])

    # Pad local tensors to max size (for scores and nodes separately)
    if local_size.item() < max_size:
        padding_size = max_size - local_size.item()
        # Pad scores with zeros, keeping type float32
        padded_scores = torch.cat([tensor_local_ppr[0], torch.zeros(padding_size, dtype=torch.float32)], dim=0)
        # Pad nodes with a dummy value (e.g., -1), keeping type long
        padded_nodes = torch.cat([tensor_local_ppr[1], torch.full((padding_size,), -1, dtype=torch.int32)], dim=0)
        tensor_local_ppr = torch.stack([padded_scores, padded_nodes], dim=0)

    # Prepare list to gather all tensors
    gather_list = [torch.zeros(2, max_size, dtype=torch.float32) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_local_ppr)

    if rank == 0:
        # Separate scores and nodes
        all_scores = torch.cat([t[0] for t in gather_list], dim=0)
        all_nodes = torch.cat([t[1] for t in gather_list], dim=0)

        # Filter out padding values (e.g., -1) from nodes
        valid_nodes_mask = all_nodes != -1
        valid_scores_mask = all_scores != 0.0

        filtered_nodes = all_nodes[valid_nodes_mask]
        filtered_scores = all_scores[valid_scores_mask]

        # Create global ppr results
        global_ppr = aggregate_ppr_results([torch.stack([filtered_scores, filtered_nodes])])
        ppr_scores, map_nodes = normalize_ppr(global_ppr)
        
        res = ppr_scores.argsort(dim = 0, descending=True)[:10]
        result_scores = {}
        for i in res:
            result_scores[map_nodes[i].item()] = ppr_scores[i].item()            
    #------------------------------------------------------------------------------------

    
    
    # Record end time
    end_time = time.time()

    # Display the results after the time recording has ended in master node only
    if rank == 0:
        display_results(config, world_size, start_time, end_time, result_scores)

    # Destroy the distributed system group
    cleanup()


def main():
    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))
    
    distributed_pagerank(rank, world_size)

if __name__ == "__main__":
    main()