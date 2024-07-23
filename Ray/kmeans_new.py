import os
import ray
from ray import tune, train
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import time
import tracemalloc
import ray._private.services

# Initialize Ray
ray.init(address = 'auto')

# Load data from a csv file - uncomment one of the data_files depending on the size you prefer
#data_file = "~/data/test_data.csv"  # 10  MB (Test)
data_file = "~/data/data_1.csv"    # 1   GB
#data_file = "~/data/data_2.csv"    # 2,5 GB
#data_file = "~/data/data.csv"      # 10  GB
df = pd.read_csv(data_file)

# Convert the DataFrame to a NumPy array
X = df.values
# Put the data in Ray's object store
X_id = ray.put(X)


def split_data(num_chunks):
    # Retrieve the data from the object store
    X_data = ray.get(X_id)
    return np.array_split(X_data, num_chunks)


@ray.remote
def ray_kmeans(data_chunk, config):
    #Uncomment to trace which worker node executes ray_kmeans each time
    """
    node_ip = ray._private.services.get_node_ip_address()
    print(f"Executing on node with IP: {node_ip}")
    """

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=config["n_clusters"], random_state=42).fit(data_chunk)

    #Compute the Calinski-Harabasz Index for performance evaluation
    calinski_harabasz = calinski_harabasz_score(data_chunk, kmeans.labels_)

    return dict(calinski_harabasz=calinski_harabasz)


def train_kmeans(config):
    # Split the data into chunks (one for each worker)
    data_chunks = split_data(config["n_chunks"])

    # Run KMeans clustering concurrently on different data chunks and different machines
    results = ray.get([ray_kmeans.options(num_cpus=4).remote(chunk, config) for chunk in data_chunks])

    # Aggregate results from all workers
    calinski_harabasz_scores = [result["calinski_harabasz"] for result in results]
    # Compute the average of the Calinski-Harabasz metric
    avg_calinski_harabasz = np.mean(calinski_harabasz_scores)


    train.report(dict(avg_calinski_harabasz=avg_calinski_harabasz))



# Record start time
# Start memory tracing
start_time = time.time()
tracemalloc.start()

# Parameters for Ray Tune - uncomment the number of chunks (workers) you prefer
config = {
        "n_clusters": 16,
        #"n_chunks": 1
        "n_chunks": 2
        }

placement_group_factory = tune.PlacementGroupFactory(
    [{"CPU": 4.0}] + [{"CPU": 4.0}] * config["n_chunks"]
)

# Ray Tune
analysis = tune.run(
    train_kmeans,
    config = config,
    num_samples = 1,  # Number of trials
    resources_per_trial = placement_group_factory,
    storage_path = "/home/user/ray/kmeans/results",
)

# Record end time
# End memory tracing
end_time = time.time()
currentMem, peakMem = tracemalloc.get_traced_memory()
tracemalloc.stop()


# Display Results

# Print results
print("\nTime taken (Ray):", end_time - start_time, "seconds")
print(f"Current memory usage is {currentMem / 10**6} MB; Peak was {peakMem / 10**6} MB\n")


# Save Results in txt file
trial = analysis.get_best_trial(metric="avg_calinski_harabasz", mode="max")
calinski_harabasz_score = trial.metric_analysis["avg_calinski_harabasz"]["max"]

results_text = (
    f"For file {data_file} and number of data chunks (worker machines) {config['n_chunks']}: \n\n"
    f"Calinski-Harabasz Score: {calinski_harabasz_score}\n"
    f"Time taken (Ray): {end_time - start_time} seconds\n"
    f"Current memory usage is {currentMem / 10**6} MB\nPeak was {peakMem / 10**6} MB\n"
)

# Create custom file name in results directory, in order to save results for different data sizes and number of machines
base_name = os.path.splitext(os.path.basename(data_file))[0]
directory = os.path.expanduser('~/ray/kmeans/results')
file_name = f"{base_name}_{config['n_chunks']}datachunks_results.txt"

results_file_name = os.path.join(directory, file_name)

# Write results to the custom text file
with open(results_file_name, "w") as f:
    f.write(results_text)


# Shutdown Ray
ray.shutdown()