import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
# tensorboard imports
from torch.utils.tensorboard import SummaryWriter
from itertools import product

def get_device_type():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    return dev


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        device_type = get_device_type()

        self.conv1 = nn.Conv2d( in_channels=1, out_channels=6, kernel_size=5).to(device_type)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5).to(device_type)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120).to(device_type)
        self.fc2 = nn.Linear(in_features=120, out_features=60).to(device_type)
        self.out = nn.Linear(in_features=60, out_features=10).to(device_type)

    def forward(self, t):
        t = t

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)

        return t

### Hyperparameter tuning 
#Alright, now we can iterate over each set of parameters using a single for-loop.
#All we have to do is unpack the set using sequence unpacking. It looks like this.

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download =True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

parameters = dict(
    lr=[0.01, 0.001],
    batch_size=[100,1000],
    shuffle=[True,False]
)

parameter_values = [v for v in parameters.values()]

network = CnnModel()

for lr, batch_size, shuffle in product(*parameter_values):
    comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    optimizer = optim.Adam(network.parameters(), lr=lr)
    
    images, labels = next(iter(train_loader))
    images = images.to(get_device_type())
    labels = labels.to(get_device_type())
    grid = torchvision.utils.make_grid(images)
    # summary writer
    tb = SummaryWriter(comment=comment)
    tb.add_image('images', grid) # this optional
    tb.add_graph(network, images)
   
    
    for epoch in range(10):
        total_correct = 0
        total_loss = 0
        
        for batch in train_loader:
            images, labels = batch # get batch
            # change compution for tensors based on dev
            images = images.to(get_device_type())
            labels = labels.to(get_device_type())
            preds = network(images) # pass batch
            
            loss = F.cross_entropy(preds, labels) # calculate the loss
            optimizer.zero_grad() # Zero graidents
            loss.backward() # calculate graidents
            optimizer.step() # update weights
            
            total_loss += loss.item() * batch_size
            total_correct += get_num_correct(preds, labels)
            
        tb.add_scalar("loss", total_loss, epoch)
        tb.add_scalar("Number correct", total_correct, epoch)
        tb.add_scalar("Accuracy", total_correct/ len(train_set), epoch)
        
        for name, param in network.named_parameters():
            tb.add_histogram(name, param, epoch)
            tb.add_histogram(f'{name}.grad', param.grad, epoch)
        
        print("epoch", epoch, "total_correct", total_correct, "loss", total_loss)
    
    tb.close()