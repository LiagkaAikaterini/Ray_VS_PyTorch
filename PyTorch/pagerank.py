import os
import time
import torch
from torch_ppr import personalized_page_rank
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import pyarrow.fs as fs
import pyarrow.csv as pv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

class GraphEdgeDataset(Dataset): 
    def __init__(self, chunk):
        self.edges = self.load_chunk(chunk)
        
    def load_chunk(self, chunk):
        nodes1 = chunk.column('node1').to_pylist()
        nodes2 = chunk.column('node2').to_pylist()
        chunk_correct_format = [nodes1, nodes2]
        
        return torch.as_tensor(data=chunk_correct_format)
        
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
def display_results(config, world_size, start_time, end_time):
    data_file = config["datafile"].split('.')[0]    # keep only data file name
    
    results_text = (
        f"\nFor file {data_file} - number of worker machines {world_size} - batch size {config['batch_size']}: \n"
        f"Time taken (Ray): {end_time - start_time} seconds\n"    
    )
    
    print(results_text)

    # Create custom file name in results directory, in order to save results for different data sizes and number of machines
    directory = os.path.expanduser('~/pytorch/pagerank/res')
    file_name = f"{data_file}_{world_size}nodes_results.txt"

    file_path = os.path.join(directory, file_name)
    
    # Write results to the custom text file
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(results_text)
    else:
        with open(file_path, 'w') as f:
            f.write(results_text)

"""
def pytorch_kmeans(data_batch, clusters):
    # perform kmeans and get the score for this data chunk
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(data_batch)

    calinski_harabasz = calinski_harabasz_score(data_batch, kmeans.labels_)

    return calinski_harabasz
"""
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
    
    # discard first and last nodes that may exist in other batches as well
    first_element = nodes1[0]
    last_element = nodes1[-1]
    
    nodes = set(nodes1)
    nodes.discard(first_element)
    nodes.discard(last_element)
    
    if len(nodes)==0:
        chosen_node_idx = None

    #print(len(nodes))
    else:
        chosen_node = list(nodes)[0]
        chosen_node_idx = node_map[chosen_node]

    return chosen_node_idx, torch.as_tensor(data= [mapped_nodes1, mapped_nodes2]), all_nodes


def result_format(nodes_list, node, pagerank_scores):
    res_dict = {}
    scores = pagerank_scores.tolist()[0]
    
    for idx, score in enumerate(scores):
        if score == 0.0:
            continue
        res_dict[nodes_list[idx]] = score
        
    return {node : res_dict}
        
    
def distributed_pagerank(rank, world_size):
    # Uncomment only the datafile you want to use
    #datafile = "test_data.csv"  # 10   MB
    datafile = "data_1.csv"     # 1    GB
    #datafile = "data_2.csv"     # 2.6  GB
    #datafile = "data.csv"       # 10.5 GB
    
    config = {
        "datafile" : datafile,
        "batch_size" : 1024 * 1024 ,  # 1MB chunks - Adjust as needed
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
        
        results = {}
        for chunk in csv_reader:
            # create the dataset of the 50MB chunks - sampler - dataloader 
            dataset = GraphEdgeDataset(chunk)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle = False)
            dataloader = DataLoader(dataset, batch_size=1024, sampler=sampler, shuffle = False)

            # each batch is approximately 10KB - 1024 samples
            for batch in dataloader:
                batch = batch.t()
                chosen_node, input1, all_nodes = get_node_of_interest(batch)
                if chosen_node==None:
                    continue
                else :
                    pagerank_results = personalized_page_rank(edge_index = input1, indices=[chosen_node])
                result = result_format(all_nodes, chosen_node, pagerank_results)
                results.update(result)
    """
    # calculate the average score in each machine
    avg_calinski_harabasz = np.mean(results)
    avg_calinski_harabasz_tensor = torch.tensor(avg_calinski_harabasz, dtype=torch.float32)

    # Gather the averages from all the machines
    gathered_results = [torch.tensor(0.0) for _ in range(world_size)]
    dist.all_gather(gathered_results, avg_calinski_harabasz_tensor)
    
    # Calculate the total average calinski-harabasz score in master node
    if rank == 0:
        global_avg_calinski_harabasz = np.mean([t.item() for t in gathered_results])
    """
    # Record end time
    end_time = time.time()

    # Display the results after the time recording has ended in master node only
    if rank == 0:
        display_results(config, world_size, start_time, end_time)

    # Destroy the distributed system group
    cleanup()


def main():
    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))
    
    distributed_pagerank(rank, world_size)

if __name__ == "__main__":
    main()