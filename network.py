import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

def get_device_type():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    return dev

def get_num_correct(preds, labels):
     return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        device_type = get_device_type()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5).to(device=device_type)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5).to(device=device_type)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120).to(device=device_type)
        self.fc2 = nn.Linear(in_features=120, out_features=60).to(device=device_type)
        self.out = nn.Linear(in_features=60, out_features=10).to(device=device_type)
    def forward(self, t):
        #(1) input layer
        t = t

        #(2) hidden layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #(3) hidden layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #(4) hidden layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        #(5) hidden layer
        t = self.fc2(t)
        t = F.relu(t)

        #(6) output layer
        t = self.out(t)

        return t

network = Network()

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

total_loss = 0
total_correct = 0
for epoch in range(5):
        
    for batch in train_loader: # get batch
        images, labels = batch
        
        images = images.to(get_device_type())
        labels = labels.to(get_device_type())

        preds = network(images) # pass batch
        loss = F.cross_entropy(preds, labels) # calcluate loss
        
        optimizer.zero_grad() # calcaute graidents
        loss.backward() # update weights
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch ", epoch, "total_correct", total_correct, "loss", total_loss)
