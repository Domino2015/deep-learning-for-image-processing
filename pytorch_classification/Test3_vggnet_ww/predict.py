
import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils

from tqdm import tqdm

from model import vgg16

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                         transform=transform)

    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_num, shuffle=True,
                                              num_workers=0)

    print("using {} images for test.".format(test_num))

    # create model
    model = vgg16(num_classes=5).to(device)
    # load model weights
    weights_path = "./VGG16.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    acc = 0.0  # accumulate accurate number / epoch

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in class_indict.values()}
    total_pred = {classname: 0 for classname in class_indict.values()}


    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            # collect the correct predictions for each class
            # value =list(class_indict.values())
            # keys = list(class_indict.keys())
            for label, prediction in zip(test_labels, predict_y):
                if label == prediction:
                    correct_pred[list(class_indict.values())[label]] += 1
                total_pred[list(class_indict.values())[label]] += 1
    # val_accurate = acc / test_num
    y_true=test_labels.cpu().numpy()
    y_pred=predict_y.cpu().numpy()
    accuracy_score_1 = accuracy_score(test_labels.cpu().numpy(), predict_y.cpu().numpy())
    precision_score_1 = precision_score(test_labels.cpu().numpy(), predict_y.cpu().numpy(),average='macro')
    recall_score_1 = recall_score(test_labels.cpu().numpy(), predict_y.cpu().numpy(),average='macro')
    f1_score_1 = f1_score(test_labels.cpu().numpy(), predict_y.cpu().numpy(),average='macro')

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    auc_1=auc(fpr, tpr)

    cm = confusion_matrix(y_true, y_pred)

    # y_scores = np.array([0.63, 0.53, 0.36, 0.02, 0.70, 1, 0.48, 0.46, 0.57])
    # y_true = np.array(['0', '1', '0', '0', '1', '1', '1', '1', '1'])
    # roc_auc_score(y_true, y_scores)

    # roc_auc_score_1 = roc_auc_score()

    print(f'Accuracy_testImages: {100 * acc // test_num} %')
    print("accuracy_score:{:.4f}  precision_score:{:.4f}  recall_score:{:.4f}  f1_score:{:.4f}  auc:{:.4f} ".format(accuracy_score_1,
                                                precision_score_1,
                                                recall_score_1,
                                                f1_score_1,
                                                auc_1))
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print('Finished!!!')



if __name__ == '__main__':
    main()

