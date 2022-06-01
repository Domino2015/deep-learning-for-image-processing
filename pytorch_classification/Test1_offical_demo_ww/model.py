import torch.nn as nn
import torch.nn.functional as F


class LeNet_ww(nn.Module):
    def __init__(self):
        super(LeNet_ww, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)



    def forward(self, x):
        x=F.relu(self.conv1(x)) # input(3, 32, 32) output(16, 14, 14)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.pool2(x)
        x=x.view(-1,5*5*16)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





# import torch
# input1=torch.rand([32,3,32,32])
# model=LeNet_ww()
# print(model)
# output=model(input1)
