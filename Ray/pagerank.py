import os
import time
import torch
from torch_ppr import page_rank
import pyarrow.fs as fs
import pyarrow.csv as pv
import ray

# Initialize Ray
ray.init()

# Function to determine the number of active nodes in ray
def get_num_nodes():
    nodes = ray.nodes()
    return sum(1 for node in nodes if node['Alive'])

def display_results(config, start_time, end_time):
    data_file = config["datafile"].split('.')[0]    # keep only data file name
    
    results_text = (
        f"\nFor file {data_file} - number of worker machines {config['num_nodes']} - batch size {config['batch_size']}: \n"
        f"Time taken (Ray): {end_time - start_time} seconds\n"    
    )
    
    print(results_text)

    directory = os.path.expanduser('~/pytorch/pagerank/res')
    file_name = f"{data_file}_{config['num_nodes']}nodes_results.txt"
    file_path = os.path.join(directory, file_name)
    
    if not os.path.exists(directory):
        # Create a new directory because it does not exist
        os.makedirs(directory)

    
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(results_text)
    else:
        with open(file_path, 'w') as f:
            f.write(results_text)

def get_node_of_interest(nodes1, nodes2):
    all_nodes_set = set(nodes1 + nodes2)
    all_nodes = list(all_nodes_set)
    all_nodes.sort()
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    
    mapped_nodes1 = [node_map[node] for node in nodes1]
    mapped_nodes2 = [node_map[node] for node in nodes2]
    
    return torch.tensor([mapped_nodes1, mapped_nodes2]), all_nodes

def result_format(nodes_list, pagerank_scores):
    res_dict = {}
    scores = pagerank_scores.tolist()
    
    for idx, score in enumerate(scores):
        if score == 0.0:
            continue
        res_dict[nodes_list[idx]] = score
        
    return res_dict

def aggregate_ppr_results(local_ppr_list):
    global_ppr = {}
    for local_ppr in local_ppr_list:
        for node, ppr_score in local_ppr.items():
            if node in global_ppr:
                global_ppr[node] += ppr_score
            else:
                global_ppr[node] = ppr_score
    return global_ppr

def normalize_ppr(global_ppr):
    total_score = sum(global_ppr.values())
    for node in global_ppr:
        global_ppr[node] /= total_score
    return global_ppr

@ray.remote
def process_chunk(data_chunk):
    nodes1 = data_chunk.column('node1').to_pylist()
    nodes2 = data_chunk.column('node2').to_pylist()

    input1, all_nodes = get_node_of_interest(nodes1, nodes2)
    pagerank_results = page_rank(edge_index=input1)
    
    result = result_format(all_nodes, pagerank_results)
    
    return result

def distributed_pagerank():
    # Uncomment only the datafile you want to use
    #datafile = "test_data.csv"  # 10   MB
    datafile = "data_1.csv"     # 1    GB
    #datafile = "data_2.csv"     # 2.6  GB
    #datafile = "data.csv"       # 10.5 GB
    
    config = {
        "num_nodes" : get_num_nodes(),
        "datafile" : datafile,
        "batch_size" : 1024 * 1024 * 50,  # 50MB chunks
        "hdfs_host" : '192.168.0.1',
        "hdfs_port" : 50000
    }

    start_time = time.time()
    
    hdfs = fs.HadoopFileSystem(host=config["hdfs_host"], port=config["hdfs_port"])
    file_to_read = f'/data/{config["datafile"]}'
    
    with hdfs.open_input_file(file_to_read) as file:
        read_options = pv.ReadOptions(block_size=config["batch_size"])
        csv_reader = pv.open_csv(file, read_options=read_options)
        
        chunk_futures = []
        results = ray.get([process_chunk.remote(batch) for batch in csv_reader])
        
        global_ppr = aggregate_ppr_results(results)
        normalized_ppr = normalize_ppr(global_ppr)
    
    end_time = time.time()
    display_results(config, len(chunk_futures), start_time, end_time)

def main():
    distributed_pagerank()

if __name__ == "__main__":
    main()

    