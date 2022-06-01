import torch
import torchvision
import torch.nn as nn
from model import LeNet_ww
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = torchvision.datasets.CIFAR10('../Test1_official_demo/data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)
    val_set = torchvision.datasets.CIFAR10(root='../Test1_official_demo/data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
    net = LeNet_ww()
    loss_function = nn.CrossEntropyLoss()
    # 第一个参数 可学习训练的所有参数
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times 迭代五次
        # 叠加的损失
        running_loss = 0.0
        # 遍历训练数据
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients 手动将梯度清零
            # XXXX: 如果不清除历史梯度，就会对计算的历史梯度进行累加（通过这个特性你能够变相实现一个很大batch数值的训练）
            optimizer.zero_grad()
            # forward + backward + optimize
            # 输入图像和标签，通过infer计算得到预测值
            outputs = net(inputs)
            # 计算损失函数；
            loss = loss_function(outputs, labels)
            # 反向传播，计算当前梯度；
            loss.backward()
            # 根据梯度更新网络参数
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                # 上下文管理器 在接下来的迭代循环中 不要去计算除了500之外的数值，不然会消耗更多的内存和硬盘
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    # 得到网络预测最大概率的输出 dim=1表示我们在第二个维度寻找最大（dim=0是batch） [1]表示提取index索引就好
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    # running_loss / 500 计算500步平均训练误差
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')
    # 保存网络和参数
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
