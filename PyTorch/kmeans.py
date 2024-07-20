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
from torch_kmeans import KMeans


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group - gloo == cpu not gpu
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
    
def create_model(rank):
    model = KMeans(n_clusters=4).to(rank)
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


def main():
    world_size = 1  # Number of GPUs/machines
    filepath = 'data/test_data.txt'
    #mp.spawn(train, args=(world_size, filepath), nprocs=world_size, join=True)
    dataset = GraphEdgeDataset(filepath)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=0)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for batch in dataloader:
        print(KMeans(batch))
        break

if __name__ == "__main__":
    main()