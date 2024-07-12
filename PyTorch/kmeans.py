import torch

# convert  tensor_data
def edges_to_tensor(edges):
    # Step 1: Extract unique nodes
    nodes = list(set([n1 for n1, n2 in edges] + [n2 for n1, n2 in edges]))

    # Step 2: Create mapping from nodes to indices
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Step 3: Convert edge list to tensor indices
    edge_indices = torch.tensor([[node_to_index[n1], node_to_index[n2]] for n1, n2 in edges], dtype=torch.long)
    
    return edge_indices, node_to_index

def kmeans(tensor_data, num_centroids, num_iterations=100):
    # Initialize centroids randomly
    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:num_centroids]]

    for _ in range(num_iterations):
        # Calculate distances from data points to centroids
        distances = torch.cdist(tensor_data, centroids)

        # Assign each data point to the closest centroid
        _, labels = torch.min(distances, dim=1)

        # Update centroids by taking the mean of data points assigned to each centroid
        for i in range(num_centroids):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(tensor_data[labels == i], dim=0)

    return centroids, labels