import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        # ToTensor()将图像转换成为 Tensor
        # Normalize（）标准化 Tensor
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # transform=transform 对图像进行预处理
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    # 将下载好的数据 分批次
    # batch_size 批次数目 shuffle 是否打乱顺序 num_workers 载入数据的线程数（windows下只能设置为0）
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    # batch_size=5000
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)

    # 转换成为一个可以迭代的迭代器 这样就可以使用 next() 来迭代的获取到 每一批次的测试图片
    # get some random training images
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()


    # 查看数据集中的图片数据
    #   画图包
    # import matplotlib.pyplot as plt
    # import numpy as np
    # # functions to show an image
    # def imshow(img):
    #     # 反标准化过程 如之前 ToTensor()函数处理时需要做标准化处理
    #     img = img / 2 + 0.5  # unnormalize
    #     # 转换为numpy格式
    #     npimg = img.numpy()
    #     # 由于在转换成Tensor时 顺序变成了【batch，channel，height，width】现在需要转换回【batch，height，width，channel】
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(val_image))
    #
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # # print labels
    # print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(100)))


    net = LeNet()
    # 交叉熵损失 这个函数已经包括了 torch.nn.LogSoftmax   torch.nn.NLLLoss
    loss_function = nn.CrossEntropyLoss()
    # 第一个参数 可学习训练的所有参数
    optimizer = optim.Adam(net.parameters(), lr=0.001)



# 训练过程
    for epoch in range(5):  # loop over the dataset multiple times 迭代五次
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
