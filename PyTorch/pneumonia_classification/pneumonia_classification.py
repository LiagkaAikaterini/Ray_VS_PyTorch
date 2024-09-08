import os
import io
import time
import torch
from torch import nn,optim
from torchvision import transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pyarrow.fs as fs
from PIL import Image

TEST = 'test'
TRAIN = 'train'
VAL ='val'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.1'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group - gloo == cpu not gpu
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class customDataset(Dataset):
    def __init__(self, config, data_dir, transform=None):
        self.hdfs = fs.HadoopFileSystem(host=config['hdfs_host'], port=config['hdfs_port'])
        self.data_dir = data_dir
        self.transform = transform
        self.file_list, self.classes, self.class_to_idx = self._get_file_list_and_classes()

    def _get_file_list_and_classes(self):
        file_list = []
        classes = []
        class_to_idx = {}
        
        # List all files and directories in the directory
        file_infos = self.hdfs.get_file_info(fs.FileSelector(self.data_dir, recursive=True))

        # add files (images) to the file_list and the directories in the classes
        for file_info in file_infos:
            if file_info.type == fs.FileType.File:
                file_list.append(file_info.path)
            elif file_info.type == fs.FileType.Directory:
                classes.append(os.path.basename(file_info.path))
        
        # attribute a unique int label to each class
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        return file_list, classes, class_to_idx

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Extract class name from the parent directory of the file
        class_name = os.path.basename(os.path.dirname(file_path))
        # Convert the class name to a numerical label (0 or 1)
        label = self.class_to_idx[class_name]
        
        # Open the image file from HDFS
        with self.hdfs.open_input_file(file_path) as f:
            img = Image.open(io.BytesIO(f.read())).convert('RGB')
        
        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label


class classify(nn.Module):
    def __init__(self,num_classes=2):
        super(classify,self).__init__()
        
         
        self.conv1=nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()        
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        self.fc=nn.Linear(in_features=32 * 112 * 112,out_features=num_classes)
        
       #Feed forward function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)            
        output=output.view(-1,32*112*112)
        output=self.fc(output)
            
        return output


def data_transforms(phase = None):
    
    if phase == TRAIN:

        data_T = T.Compose([
            
                T.Resize(size = (256,256)),
                T.RandomRotation(degrees = (-20,+20)),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    elif phase == TEST or phase == VAL:

        data_T = T.Compose([

                T.Resize(size = (224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    return data_T
    

# Display and save the results
def display_results(world_size, start_time, end_time, result_text):
    
    results_text = (
        f"\nPneumonia Classification - Number of worker machines {world_size} : \n\n"
        f"Time taken (PyTorch): {end_time - start_time} seconds\n"    
    ) + result_text
    
    print(results_text)

    # Create custom file name in results directory, in order to save results for different number of machines
    directory = os.path.expanduser('~/PyTorch/pneumonia_classification/res')
    file_name = f"{world_size}nodes_results.txt"

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


def train(trainloader, optimizer, model, criterion):
    model.train()   # Set model to training mode
    running_loss = 0
    for images, labels in trainloader:
        
        '''
        #Changing images to cuda for gpu
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        '''
        
        # Training pass
        # Sets the gradient to zero
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        # This is where the model learns by backpropagating
        # accumulates the loss for mini batch
        loss.backward()
        
        # And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss
    
            
def test(testloader, device, model):
    model.eval()  # Set model to evaluation mode
    correct_count, all_count = 0, 0
    for images,labels in testloader:
        for i in range(len(labels)):
            '''
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            '''
            
            images, labels = images.to(device), labels.to(device)
            
            img = images[i].view(1, 3, 224, 224)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.cpu()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
        
    return all_count, correct_count


def distributed_classification(rank, world_size, config):    
    # Record start time
    start_time = time.time()
    
    # Create the distributed system group
    # with the number of machines that is defined from the torchrun command
    setup(rank, world_size)
    
    result_text = ""

    # create the datasets - samplers - dataloaders
    trainset = customDataset(config = config, data_dir = os.path.join(config['data_dir'], TRAIN), transform = data_transforms(TRAIN))
    testset = customDataset(config = config, data_dir = os.path.join(config['data_dir'], TEST), transform = data_transforms(TEST))
    validset = customDataset(config = config, data_dir = os.path.join(config['data_dir'], VAL), transform = data_transforms(VAL))
    
    
    # get info for the results
    class_names = trainset.classes
    result_text += f'\nClass Names : {class_names}\n'
    result_text += f'Class to index : {trainset.class_to_idx}\n\n'
    
    
    trainsampler = DistributedSampler(trainset, num_replicas = world_size, rank = rank, shuffle = True)
    testsampler = DistributedSampler(testset, num_replicas = world_size, rank = rank, shuffle = True)
    validsampler = DistributedSampler(validset, num_replicas = world_size, rank = rank, shuffle = True)


    trainloader = DataLoader(trainset, batch_size = config['batch_size'], sampler = trainsampler)
    validloader = DataLoader(validset, batch_size = config['batch_size'], sampler = validsampler)
    testloader = DataLoader(testset, batch_size = config['batch_size'], sampler = testsampler)

    # get info for the results
    images, labels = next(iter(trainloader))

    result_text += f'Images Shape : {images.shape}\n'
    result_text += f'Labels Shape : {labels.shape}\n\n'

    '''
    for i, (images,labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
    '''          

    images.shape, labels.shape
    
    # in our case gpu in not available so we will do CPU-only training 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
    if torch.cuda.is_available():
        summary(classify().cuda(), (images.shape[1], images.shape[2], images.shape[3]))
    '''
    
    # Model initialization
    model = classify().to(device)
    model = DDP(model)
    # for gpu
    #model = DDP(model, device_ids=[rank])
    
    # defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr = config['lr'])
    
    # defining the loss function
    criterion = nn.CrossEntropyLoss()
    
    '''
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    '''    
    
    # Train  
    for i in range(config['epochs']):
        # set epoch to the sampler for synchronization in all the nodes
        trainsampler.set_epoch(i+1)
        
        running_loss = train(trainloader, optimizer, model, criterion)
        
        # Reduce running loss across all processes
        gathered_running_loss = torch.tensor(running_loss, dtype=torch.float64)
        dist.all_reduce(gathered_running_loss, op=dist.ReduceOp.SUM)
        
        # get the average running loss across all the machines
        gathered_running_loss /= world_size
        avg_running_loss = gathered_running_loss.item()
        
        # add to result_text only in master machine
        if rank == 0:
            result_text += f'Epoch {i+1} - Training loss: {avg_running_loss/len(trainloader)}\n'
            print(f'Epoch {i+1} - Training loss: {avg_running_loss/len(trainloader)}')
        
            
            
    # Test
    all_count, correct_count = test(testloader, device, model)

    # Gather the results
    gathered_all_count = torch.tensor(all_count, dtype=torch.int64)
    gathered_correct_count = torch.tensor(correct_count, dtype=torch.int64)
    dist.all_reduce(gathered_all_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(gathered_correct_count, op=dist.ReduceOp.SUM)
    
    # add to result_text only in master machine
    if rank == 0 :
        result_text += f'Number Of Images Tested = {gathered_all_count} \n'
        result_text += f'\nModel Accuracy = {gathered_correct_count/gathered_all_count} \n\n'
        
    # Record end time
    end_time = time.time()

    # Display the results after the time recording has ended only in the master node
    if rank == 0:
        display_results(world_size, start_time, end_time, result_text)

    # Destroy the distributed system group
    cleanup()

            
def main():
    
    config = {
        'hdfs_host' : '192.168.0.1',
        'hdfs_port' : 50000,
        'lr': 0.01, 
        'batch_size': 64, 
        'epochs': 10,
        'data_dir' : "/data/chest_xray"
    }
    
    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))
    
    distributed_classification(rank, world_size, config)


if __name__ == "__main__":
    main()