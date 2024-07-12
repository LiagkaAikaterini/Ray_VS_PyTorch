import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
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


def create_dataloader(rank, world_size, filepath):
    dataset = GraphEdgeDataset(filepath)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    return dataloader



def train(rank, world_size):
    setup(rank, world_size)

    model = create_model(rank)
    dataloader = create_dataloader(rank, world_size)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):  # Number of epochs
        dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        dist.barrier()

    cleanup()
    
    
def main():
    world_size = 3  # Number of GPUs/machines
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()