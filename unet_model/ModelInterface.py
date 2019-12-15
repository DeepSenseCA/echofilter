import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision

from DeepSense.model.UNet import UNet


class ModelInterface:

    def __init__(self):
        self.net = UNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def train_unet(self, train_loader):
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        PATH = './cifar_net.pth'
        torch.save(self.net.state_dict(), PATH)
        print('Finished Training')


