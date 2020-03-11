import torchvision
import torchvision.transforms as transforms
from finalModel import finalModel
from RunBuilder import RunBuilder
from RunManager import RunManager
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils import get_device_type


def train():

    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    params = OrderedDict(
        lr=[0.01, 0.001],
        batch_size=[100, 1000],
        shuffle=[True, False],
        num_workers=[2]
    )

    runManager = RunManager()

    for run in RunBuilder.get_runs(params):

        network = finalModel()
        loader = DataLoader(
            train_set, batch_size=run.batch_size, shuffle=run.shuffle, num_workers=run.num_workers)

        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        runManager.begin_run(run, network, loader)

        for epoch in range(10):
            runManager.begin_epoch()
            for batch in loader:

                images, labels = batch
                # support computation based on device type
                images = images.to(get_device_type())
                labels = labels.to(get_device_type())

                preds = network(images)
                loss = F.cross_entropy(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                runManager.track_loss(loss)
                runManager.track_num_correct(preds, labels)

            runManager.end_epoch()

        runManager.end_run()
    runManager.save('results')


if __name__ == "__main__":
    train()
