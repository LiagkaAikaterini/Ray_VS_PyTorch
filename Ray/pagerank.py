import os
import time
import torch
from torch_ppr import page_rank
import pyarrow.fs as fs
import pyarrow.csv as pv
import ray
import json
import gc

# Function to determine the number of active nodes in ray
def get_num_nodes():
    nodes = ray.nodes()
    return sum(1 for node in nodes if node['Alive'])

# Displays and saves the results
def display_results(config, start_time, end_time, scores_dict):
    data_file = config["datafile"].split('.')[0]    # keep only data file name

    results_text = (
        f"\nFor file {data_file} - number of worker machines {config['num_nodes']} - batch size {config['batch_size']}: \n"
        f"Time taken (Ray): {end_time - start_time} seconds\n"
        f"Top 10 scores: {scores_dict} \n"
    )

    print(results_text)
    
    # Create custom file name in results directory, in order to save results for different data sizes and number of machines
    directory = os.path.expanduser('~/ray/pagerank/res')
    file_name = f"{data_file}_{config['num_nodes']}nodes_results.txt"

    file_path = os.path.join(directory, file_name)
    
    # Create a new directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    
    # Write results to the custom text file - keep the previous results if they exist in the file
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(results_text)
    else:
        with open(file_path, 'w') as f:
            f.write(results_text)


# Creates node map and transform data in the right format for torch_ppr pagerank
def input_format(nodes1, nodes2):
    # create set to get rid of duplicates
    all_nodes_set = set(nodes1 + nodes2)
    
    # create node dictionary to know which node corresponds to which index
    all_nodes = list(all_nodes_set)
    all_nodes.sort()
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    
    # swap node ids with consecutive indexes starting from 0
    mapped_nodes1 = [node_map[node] for node in nodes1]
    mapped_nodes2 = [node_map[node] for node in nodes2]
    
    return torch.tensor([mapped_nodes1, mapped_nodes2]), all_nodes

# helper function to add nodes and corresponding scores to existing score dictionary
def add_to_dict(global_pr, scores, nodes):
    for idx, node in enumerate(nodes):
        if node in global_pr:
            global_pr[node] += scores[idx]
        else:
            global_pr[node] = scores[idx]
                
    return global_pr


# adds dictionary to existing score dictionary
def aggregate_pr_results(global_pr_dict, new_pr_dict):
    for node, score in new_pr_dict.items():
        if node in global_pr_dict:
            global_pr_dict[node] += score
        else:
            global_pr_dict[node] = score
            
    return global_pr_dict

# gets the N higher scores from scores dictionary
def top_scores(N, scores_dict):
    res = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[:N]
    top_N_scores = dict(res)
    
    return top_N_scores

# saves the intermediate dictionary results in files in order to relief memory
def save_intermediate_results(intermediate_result, filename):
    directory = os.path.expanduser('~/ray/pagerank/intermediate_results')
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)

    with open(file_path, 'w') as f:
        json.dump(intermediate_result, f)


# loads all the intermediate dictionaries in one dictionary to get the real, aggregated result
def load_intermediate_results():
    top_aggregated_result = {}
    directory = os.path.expanduser('~/ray/pagerank/intermediate_results')

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

# normalizes the scores to all sum to 1
def normalize_pr(scores_dict):
    # Extract scores from the dictionary
    scores = list(scores_dict.values())
    
    # Calculate the total score
    total_score = sum(scores)
    
    # Normalize scores by dividing by the total score
    normalized_scores = [score / total_score for score in scores]
    
    # Reconstruct the dictionary with normalized scores
    normalized_scores_dict = {node: score for node, score in zip(scores_dict.keys(), normalized_scores)}
    
    return normalized_scores_dict

# Deletes all the files used to save the intermediate results
def cleanup():
    directory =  os.path.expanduser('~/ray/pagerank/intermediate_results')
    if os.path.exists(directory):
        # List all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

@ray.remote
def pagerank(data_chunk):
    nodes1 = data_chunk.column('node1').to_pylist()
    nodes2 = data_chunk.column('node2').to_pylist()

    pr_input, all_nodes = input_format(nodes1, nodes2)
    pagerank_results = page_rank(edge_index=pr_input)
    
    res = add_to_dict({}, pagerank_results.tolist(), all_nodes)

    return res


def distributed_pagerank(config):
    # Read the data from hdfs in managable chunks
    hdfs = fs.HadoopFileSystem(host=config["hdfs_host"], port=config["hdfs_port"])
    file_to_read = f'/data/{config["datafile"]}'
    
    with hdfs.open_input_file(file_to_read) as file:
        read_options = pv.ReadOptions(block_size=config["batch_size"])
        csv_reader = pv.open_csv(file, read_options=read_options)
       
        futures = []
        global_pr = {}
        for i, chunk in enumerate(csv_reader):
            print("\n\nChunk", i)
            futures.append(pagerank.remote(chunk))
            # every ten futures execute the distributed calculations
            if i % 10 == 0:
                batch_results = [ray.get(f) for f in futures]
                # gather all the results in a dictionary
                global_pr = {}
                for scores_dict in batch_results:
                    global_pr = aggregate_pr_results(global_pr, scores_dict)

                futures = []
                batch_results = []
                gc.collect()
                
                if i % 30 == 0:
                    # After aggregating results save them in an intermediate file for memory relief
                    save_intermediate_results(global_pr, f"result_chunk_{i}.json")
                
                    # Clear current results to free memory
                    global_pr = {}
                    gc.collect()
    
    # make sure the last batch results are calculated (even if i % 10 != 0)
    if futures != [] :
        batch_results = [ray.get(f) for f in futures]
        futures = []
        
        global_pr = {}
        for scores_dict in batch_results:
            global_pr = aggregate_pr_results(global_pr, scores_dict)
        
        save_intermediate_results(global_pr, f"result_chunk_last.json")
        
        global_pr = {}
        batch_results = []
        
        gc.collect()
    
    # aggregate the results (the 10.000 top scores)
    final_aggregated_result = load_intermediate_results()
    # normalize the results
    normalized_pr = normalize_pr(final_aggregated_result)
    gc.collect()
    
    # get the top 10 scores to display
    result_scores = top_scores(10, normalized_pr)
    
    return result_scores
    

def main():   
    # Record start time
    start_time = time.time()
    
    # Initialize Ray
    ray.init(address='auto')
    
    # Uncomment only the datafile you want to use
    #datafile = "test_data.csv"  # 10   MB
    datafile = "data_1.csv"     # 1    GB
    #datafile = "data_2.csv"     # 2.6  GB
    #datafile = "data.csv"       # 10.5 GB
    
    config = {
        "num_nodes" : get_num_nodes(),
        "datafile" : datafile,
        "batch_size" : 1024 * 1024 * 10,  # 50MB chunks
        "hdfs_host" : '192.168.0.1',
        "hdfs_port" : 50000
    }
    
    # Perform the distributed pagerank
    result_scores = distributed_pagerank(config)
    
    # Record end time
    end_time = time.time()

    # Save and display the results after the time recording has ended
    display_results(config, start_time, end_time, result_scores)

    # Delete all the intermediate result files, which are no longer needed
    cleanup()
    
    # Shutdown Ray
    ray.shutdown()
    

if __name__ == "__main__":
    main()

    
