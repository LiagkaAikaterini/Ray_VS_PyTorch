import ray
from ray import tune, train
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import time
import tracemalloc

# Initialize Ray with 3 workers
ray.init(address = 'auto')  #??? ISWS KENO, ISWS ME ray.init(address='auto') CHECKARE TO

# Load data from a text file
# Assume the text file is a CSV with a header row
#data_file = "connected_graph.csv"
data_file = "test_data.csv"
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
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=config["n_clusters"], random_state=42).fit(data_chunk)

    #Compute the Calinski-Harabasz Index for performance evaluation
    calinski_harabasz = calinski_harabasz_score(data_chunk, kmeans.labels_)

    return dict(calinski_harabasz=calinski_harabasz)


def train_kmeans(config):
    # Split the data into chunks (one for each worker)
    data_chunks = split_data(config["n_chunks"])

    # Run KMeans clustering in parallel on different data chunks
    results = ray.get([ray_kmeans.options(num_cpus=4).remote(chunk, config) for chunk in data_chunks])

    # Aggregate results from all workers
    calinski_harabasz_scores = [result["calinski_harabasz"] for result in results]
    # Compute the average of each metric
    avg_calinski_harabasz = np.mean(calinski_harabasz_scores)


    train.report(dict(avg_calinski_harabasz=avg_calinski_harabasz))



# Record start time
# Start memory tracing
start_time = time.time()
tracemalloc.start()

config = {
        "n_clusters": 16,
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
end_time = time.time()

# End memory tracing
currentMem, peakMem = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Print results
print("Time taken (Ray):", end_time - start_time, "seconds")
print(f"Current memory usage is {currentMem / 10**6}MB; Peak was {peakMem / 10**6}MB")


# Shutdown Ray
ray.shutdown()