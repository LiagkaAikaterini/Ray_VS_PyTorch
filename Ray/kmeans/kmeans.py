import os
import ray
import numpy as np
import ray.data
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import time
import pyarrow.fs as fs
import pyarrow.csv as pv

    
# Function to determine the number of active nodes in ray
def get_num_nodes():
    nodes = ray.nodes()
    return sum(1 for node in nodes if node['Alive'])


# Display and save the results
def display_results(config, start_time, end_time, end_time_system, calinski_harabasz_res):
    data_file = config["datafile"].split('.')[0]    # keep only data file name
    
    results_text = (
        f"\nFor file {data_file} - number of worker machines {config['num_nodes']} - batch size {config['batch_size']}: \n\n"
        f"Calinski-Harabasz Score: {calinski_harabasz_res}\n"
        f"Time taken (Ray): {end_time - start_time} seconds\n"  
        f"Time taken for system to initialize (Ray): {end_time_system - start_time} seconds\n"    
    )
    
    print(results_text)

    # Create custom file name in results directory, in order to save results for different data sizes and number of machines
    directory = os.path.expanduser('~/Ray/kmeans/res')
    file_name = f"{data_file}_{config['num_nodes']}nodes_results.txt"

    file_path = os.path.join(directory, file_name)
    
    # Create a new directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write results to the custom text file
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(results_text)
    else:
        with open(file_path, 'w') as f:
            f.write(results_text)


@ray.remote
def ray_kmeans(data_batch, config):
    # get which node executes this function each time for monitoring reasons
    node_ip = ray._private.services.get_node_ip_address()
    print(f"Executing on node with IP: {node_ip}")

    # perform kmeans and get the score for this data chunk
    kmeans = KMeans(n_clusters=config["n_clusters"], random_state=42)
    kmeans.fit(data_batch)

    calinski_harabasz = calinski_harabasz_score(data_batch, kmeans.labels_)

    return calinski_harabasz


def distributed_kmeans(config):
    hdfs_host = '192.168.0.1'
    hdfs_port = 50000

    
    # Connect to HDFS using PyArrow's FileSystem
    hdfs = fs.HadoopFileSystem(host=hdfs_host, port=hdfs_port)
    file_to_read = f'/data/{config["datafile"]}'

    with hdfs.open_input_file(file_to_read) as file:
        # Define CSV read options to read in chunks
        read_options = pv.ReadOptions(block_size=config["batch_size"])  # 50 MB chunks
        csv_reader = pv.open_csv(file, read_options=read_options)
        
        results = ray.get([ray_kmeans.remote(batch.to_pandas().values, config) for batch in csv_reader])
    
    avg_calinski_harabasz = np.mean(results)

    return avg_calinski_harabasz


def main():
    # Record start time
    start_time = time.time()
    
    # Initialize Ray
    ray.init(address='auto')
    
    # Record time for system initialization
    end_time_system = time.time()
    
    # Uncomment only the datafile you want to use
    #datafile = "test_data.csv"  # 10   MB
    datafile = "data_1.csv"     # 1    GB
    #datafile = "data_2.csv"     # 2.6  GB
    #datafile = "data.csv"       # 10.5 GB
    
    # Parameters
    config = {
        "datafile" : datafile,
        "n_clusters": 16,
        "num_nodes" : get_num_nodes(),
        "cpus_per_node" : 4,
        "batch_size" : 1024 * 1024 * 20  # 20 MB chunks - Adjust as needed
    }

    # Perform the distributed kmeans
    res_score = distributed_kmeans(config)
 
    # Record end time
    end_time = time.time()

    # save the results
    display_results(config, start_time, end_time, end_time_system, res_score)

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
