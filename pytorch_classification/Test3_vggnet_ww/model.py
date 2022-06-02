import torch.nn as nn
import torch
import torch.nn.functional as F


class vgg16(nn.Module):
    def __init__(self,num_classes=5, init_weights=False):
        super(vgg16, self).__init__()
        self.conv1=nn.Conv2d(3,64,3,stride=1,padding=1)
        self.conv1_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))  # input(3, 224, 224) output(64, 224, 224)
        x = F.relu(self.conv1_1(x))  # input(64, 224, 224) output(64, 224, 224)
        x = self.pool1(x)   # input(64, 224, 224) output(64, 112, 112)

        x = F.relu(self.conv2(x))   # input(64, 112, 112) output(128, 112, 112)
        x = F.relu(self.conv2_1(x))   # input(128, 112, 112) output(128, 112, 112)
        x = self.pool2(x)  # input(128, 112, 112) output(128, 56, 56)


        x = F.relu(self.conv3(x))  # input(128, 56, 56) output(256, 56, 56)
        x = F.relu(self.conv3_1(x))  # input(256, 56, 56) output(256, 56, 56)
        x = F.relu(self.conv3_1(x))      # input(256, 56, 56) output(256, 56, 56)
        x = self.pool3(x)   # input(256, 56, 56) output(256, 28, 28)

        x = F.relu(self.conv4(x))  # input(256, 28, 28) output(512, 28, 28)
        x = F.relu(self.conv4_1(x))  # input(512, 28, 28) output(512, 28, 28)
        x = F.relu(self.conv4_1(x))  # input(512, 28, 28) output(512, 28, 28)
        x = self.pool4(x)  # input(512, 28, 28) output(512, 14, 14)

        x = F.relu(self.conv5(x))  # input(512, 14, 14) output(512, 14, 14)
        x = F.relu(self.conv5(x))  # input(512, 14, 14) output(512, 14, 14)
        x = F.relu(self.conv5(x))  # input(512, 14, 14) output(512, 14, 14)
        x = self.pool5(x)  # input(512, 14, 14) output(512, 7, 7)

        x = x.view(-1, 7 * 7 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # 判断层的名称
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# import torch
# input1=torch.rand([32,3,224,224])
# model=vgg16()
# print(model)
# output=model(input1)
# print(output)