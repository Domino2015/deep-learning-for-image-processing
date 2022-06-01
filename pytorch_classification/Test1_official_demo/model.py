import torch.nn as nn
import torch.nn.functional as F

# 首先，定义一个类，继承于nn.Module
class LeNet(nn.Module):
    # 初始化各层网络的结构、参数
    # 换句话说，就是定义整个网络的基础模块
    def __init__(self):
        # super（）解决多重继承中，解析顺序中调用正确的下一个父类函数
        # 不需要您明确引用 父/类名称
        super(LeNet, self).__init__()
        # (in_channels, out_channels（卷积核个数）, kernel_size）
        self.conv1 = nn.Conv2d(3, 16, 5)
        # 池化核大小2 步长2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 通过第一个卷积层 深度（卷积核个数）已经变成16
        self.conv2 = nn.Conv2d(16, 32, 5)

        self.pool2 = nn.MaxPool2d(2, 2)
        # 第一个参数 输入维度 第二个参数 输出维度（隐藏层的神经元个数）
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        #最后一层的输出需要 根据具体分类种类数量来确定
        self.fc3 = nn.Linear(84, 10)

    # 定义正向传播的过程
    # 当实例化这个类时，传参时，安装forward顺序进行。 输入Tensor
    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        # flatten将特征向量展平（拉伸）为一维向量
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        # 在计算交叉熵的时候 已经内置 softmax函数 这里不需要额外添加
        return x




import torch
input1=torch.rand([32,3,32,32])
model=LeNet()
print(model)
output=model(input1)