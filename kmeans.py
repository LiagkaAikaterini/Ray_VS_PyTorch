import ray
from ray.util import cluster
from ray.util import graph
import time

# Initialize Ray with 3 workers
ray.init(num_cpus=3)

# Read edges from the text file
# Read edges from the text file
edges = []
with open('data/test.txt', 'r') as file:
    for line in file:
        if line.split()[0] == '#':
            continue
        edges.append(tuple(map(int, line.strip().split())))
print(edges)

# Create a graph with Ray Graph object
graph_data = graph.Graph(edges)

# Convert the graph data to a NumPy array
adjacency_array_ray = graph_data.adjacency_matrix()

# Define the k-means parameters
num_clusters = 1
num_iterations = 100

# Record start time
start_time = time.time()

# Perform distributed k-means clustering using Ray
centroids, cluster_assignments_ray = cluster.kmeans(adjacency_array_ray, num_clusters, num_iterations)

# Record end time
end_time = time.time()

# Print results
print("Distributed K-Means Centroids (Ray):", centroids)
print("Distributed K-Means Cluster Assignments (Ray):", cluster_assignments_ray)
print("Time taken (Ray):", end_time - start_time, "seconds")

# Shutdown Ray
ray.shutdown()