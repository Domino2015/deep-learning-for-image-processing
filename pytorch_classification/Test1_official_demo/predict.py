import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('test1.jpg')
    # 转换为Tensor
    im = transform(im)  # [C, H, W]
    # 在Tensor最前面（dim=0）添加一个 batch 维度
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
    # 不需要 求损失参数 disable gradients
    with torch.no_grad():
        outputs = net(im)
        # predict = torch.softmax(outputs, dim=1) #输出概率和为1的概率分布
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
