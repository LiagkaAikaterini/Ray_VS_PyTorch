import ray
from ray import tune
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import time
import tracemalloc

# Initialize Ray with 3 workers
ray.init(address = 'auto')  #??? ISWS KENO, ISWS ME ray.init(address='auto') CHECKARE TO 

# Load data from a text file
# Assume the text file is a CSV with a header row
data_file = "data/test_data.csv"
df = pd.read_csv(data_file)

# Convert the DataFrame to a NumPy array
X = df.values

# Put the data in Ray's object store
X_id = ray.put(X)

def ray_kmeans(config):

    # Retrieve the data from the object store
    X_data = ray.get(X_id)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=config["n_clusters"], random_state=42).fit(X_data)

    # Compute the Silhouette Score
    silhouette_avg = silhouette_score(X_data, kmeans.labels_)

    #Compute the Calinski-Harabasz Index
    calinski_harabasz = calinski_harabasz_score(X_data, kmeans.labels_)

    # Compute Inertia
    inertia = kmeans.inertia_

    # Report the scores
    tune.report(silhouette_score=silhouette_avg,
                calinski_harabasz=calinski_harabasz,
                inertia=inertia)


# Record start time
start_time = time.time()

# Start memory tracing
tracemalloc.start()

# Ray Tune
analysis = tune.run(
    ray_kmeans,
    config = {"n_clusters": 16},  # Example grid search over different cluster sizes
    num_samples = 3,  # Number of trials
    resources_per_trial = {"cpu": 4},
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

# Best config - Silhouette Score
best_config = analysis.get_best_config(metric="silhouette_score", mode="max")
best_silhouette_score = analysis.get_best_trial(metric="silhouette_score", mode="max").last_result["silhouette_score"]
print("Best Configuration:", best_config)
print("Best Silhouette Score:", best_silhouette_score)

# Best config - Calinski-Harabasz Index
best_calinski_harabasz_config = analysis.get_best_config(metric="calinski_harabasz", mode="max")
best_calinski_harabasz_score = analysis.get_best_trial(metric="calinski_harabasz", mode="max").last_result["calinski_harabasz"]
print("Best Calinski-Harabasz Configuration:", best_calinski_harabasz_config)
print("Best Calinski-Harabasz Score:", best_calinski_harabasz_score)

# Best config - Inertia
best_inertia_config = analysis.get_best_config(metric="inertia", mode="min")
best_inertia = analysis.get_best_trial(metric="inertia", mode="min").last_result["inertia"]
print("Best Inertia Configuration:", best_inertia_config)
print("Best Inertia:", best_inertia)

# Shutdown Ray
ray.shutdown()