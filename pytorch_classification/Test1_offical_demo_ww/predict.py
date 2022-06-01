
import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils

from tqdm import tqdm

from model import LeNet_ww

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                         transform=transform)

    val_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=val_num, shuffle=True,
                                              num_workers=0)


    # create model
    model = LeNet_ww().to(device)
    # load model weights
    weights_path = "./LeNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
    val_accurate = acc / val_num
    print(f'Accuracy_TestImages: {100 * acc // val_num} %')
    print('Finished!!!')



if __name__ == '__main__':
    main()


