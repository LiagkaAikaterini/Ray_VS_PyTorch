import time
import tracemalloc
import torch
from torch.utils.data import Dataset
from parallelism import GraphEdgeDataset
import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.models import resnet18
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group - gloo == cpu not gpu
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
    
def create_model(rank):
    model = resnet18().to(rank)
    fsdp_model = FSDP(model, auto_wrap_policy=always_wrap_policy)
    return fsdp_model


class GraphEdgeDataset(Dataset):
    def __init__(self, filepath):
        self.edges = self.load_edges(filepath)

    def load_edges(self, filepath):
        edges = []
        with open(filepath, 'r') as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                edges.append((node1, node2))
        return edges

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        return self.edges[idx]


def train(rank, world_size, filepath):
    
    # Record start time and memory tracing
    start_time = time.time()
    tracemalloc.start()
    
    
    setup(rank, world_size)

    model = create_model(rank)
    
    dataset = GraphEdgeDataset(filepath)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    for batch in DataLoader:
        print( batch)
    
    tensor_data = edges_to_tensor(dataloader)
    
    num_centroids = 10
    #centroids, labels = dist_kmeans(rank, world_size, tensor_data, num_centroids)
    
    # Record end time and end memory tracing
    end_time = time.time()
    currentMem, peakMem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
   
    if rank == 0:
        #print("Final Centroids:\n", centroids)
        print("Time taken for K-Means: {:.4f} seconds".format(end_time - start_time))

    
    print(f"Current memory usage is {currentMem / 10**6}MB; Peak was {peakMem / 10**6}MB")

    cleanup()

# convert  tensor_data
def edges_to_tensor(edges):
    # Step 1: Extract unique nodes
    nodes = list(set([n1 for n1, n2 in edges] + [n2 for n1, n2 in edges]))

    # Step 2: Create mapping from nodes to indices
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Step 3: Convert edge list to tensor indices
    edge_indices = torch.tensor([[node_to_index[n1], node_to_index[n2]] for n1, n2 in edges], dtype=torch.long)
    
    return edge_indices, node_to_index

def dist_kmeans(rank, world_size, tensor_data, num_centroids, num_iterations=100):
    # Initialize centroids randomly
    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:num_centroids]]

    for _ in range(num_iterations):
        # Calculate distances from data points to centroids
        distances = torch.cdist(tensor_data, centroids)

        # Assign each data point to the closest centroid
        _, labels = torch.min(distances, dim=1)

        # Compute local centroid updates
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_centroids, device=tensor_data.device)
        
        # Update centroids by taking the mean of data points assigned to each centroid
        for i in range(num_centroids):
            if torch.sum(labels == i) > 0:
                new_centroids[i] = torch.sum(tensor_data[labels == i], dim=0)
                counts[i] = torch.sum(labels == i)
                
        # All-gather new centroids and counts
        dist.all_reduce(new_centroids, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        
        centroids = new_centroids / counts[:, None]

    return centroids, labels  
    
def main():
    world_size = 1  # Number of GPUs/machines
    filepath = 'data/test_data.txt'
    #mp.spawn(train, args=(world_size, filepath), nprocs=world_size, join=True)
    dataset = GraphEdgeDataset(filepath)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=0)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    for batch in dataloader:
        print(batch[0])
        break

if __name__ == "__main__":
    main()