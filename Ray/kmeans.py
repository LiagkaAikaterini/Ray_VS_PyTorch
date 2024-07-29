import os
import ray
import numpy as np
import ray.data
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import time
import tracemalloc
import pyarrow.fs as fs
import pyarrow.csv as pv

# Function to determine the number of active nodes
def get_num_nodes():
    nodes = ray.nodes()
    return sum(1 for node in nodes if node['Alive'])

def display_results(config, start_time, end_time, currentMem, peakMem, calinski_harabasz_res):
    # Display Results
    data_file = config["datafile"].split('.')[0]    # keep only data file name

    results_text = (
        f"For file {data_file} - number of worker machines {config['num_nodes']} - batch size {config['batch_size']}: \n\n"
        f"Calinski-Harabasz Score: {calinski_harabasz_res}\n"
        f"Time taken (Ray): {end_time - start_time} seconds\n"
        f"Current memory usage is {currentMem / (1024**2):.2f} MB\nPeak was {peakMem / (1024**2):.2f} mB\n\n"
    )

    print(results_text)

    # Create custom file name in results directory, in order to save results for different data sizes and number of machines
    directory = os.path.expanduser('~/ray/kmeans/res')
    file_name = f"{data_file}_{config['num_nodes']}nodes_results.txt"

    file_path = os.path.join(directory, file_name)
    
    # Write results to the custom text file
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(results_text)
    else:
        with open(file_path, 'w') as f:
            f.write(results_text)


@ray.remote
def ray_kmeans(data_batch, config):
    node_ip = ray._private.services.get_node_ip_address()
    print(f"Executing on node with IP: {node_ip}")

    kmeans = KMeans(n_clusters=config["n_clusters"], random_state=42)
    kmeans.fit(data_batch)

    calinski_harabasz = calinski_harabasz_score(data_batch, kmeans.labels_)

    return calinski_harabasz


def train_kmeans(config):
    hdfs_host = '192.168.0.1'
    hdfs_port = 50000  # Default HDFS port, change if necessary

    # Connect to HDFS using PyArrow's FileSystem
    hdfs = fs.HadoopFileSystem(host=hdfs_host, port=hdfs_port)
    hdfs_path = '/data'
    file_to_read = f'{hdfs_path}/{config["datafile"]}'

    with hdfs.open_input_file(file_to_read) as file:
        # Define CSV read options to read in chunks
        read_options = pv.ReadOptions(block_size=config["batch_size"])  # 50 MB chunks
        csv_reader = pv.open_csv(file, read_options=read_options)

        results = ray.get([ray_kmeans.remote(batch, config) for batch in csv_reader])

    #calinski_harabasz_scores = [result["calinski_harabasz"] for result in results]
    avg_calinski_harabasz = np.mean(results)

    return avg_calinski_harabasz


def main():
    # Initialize Ray
    ray.init(address='auto')

    # Record start time and memory usage
    start_time = time.time()
    tracemalloc.start()

    # Parameters
    config = {
        "datafile" : "data_1.csv",
        "n_clusters": 16,
        "num_nodes" : get_num_nodes(),
        "cpus_per_node" : 4,
        "batch_size" : 1024 * 1024 * 50  # 50 MB chunks - Adjust as needed
    }

    res_score = train_kmeans(config)
    
    """
    # Run Ray Tune
    analysis = tune.run(
        train_kmeans,
        config=config,
        num_samples=1,  # Number of trials
        resources_per_trial={"cpu": num_nodes * cpus_per_node},  # Adjust resources dynamically
        storage_path="/home/user/ray/kmeans/results",
    )
    """

    # Record end time and memory usage
    end_time = time.time()
    currentMem, peakMem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    display_results(config, start_time, end_time, currentMem, peakMem, res_score)

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
