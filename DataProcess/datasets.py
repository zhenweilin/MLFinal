import torch
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets
import sys
import os
import json,codecs
import numpy as np
# os.chdir(os.path.join(sys.path[0],'DataProcess'))
# os.chdir(sys.path[0])
#外面调用要有这句，可能是因为最开始的时候文件放在外面，是从外面移动进来的，执行目录还在外面
from mean_std import mean_std
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# sys.path.append(os.path.join(sys.path[0],'DataProcess'))

DATA_DIR = os.path.join(sys.path[0],'DataProcess/datasets')
# image_path = '/home/linzhenwei/MLFinal/DataProcess/datasets/train/NORMAL/IM-0115-0001.jpeg'

def get_mean_std(n,type):
    '''
    type is train or test
    n is 1 when picture is gray, and n is 3 when the picture is RGB
    '''
    if not os.path.exists(os.path.join(DATA_DIR,'{}.json'.format(type))):
        mean_std(n).get_mean_std('{}'.format(type))
    mean_std1 = codecs.open(os.path.join(DATA_DIR,'{}.json'.format(type)),'r').read()
    mean_std1 = json.loads(mean_std1)
    norm = np.array(mean_std1)
    norm = dict(np.ndenumerate(norm))
    return norm


def Datasets(batch_size = 128):
    norm_train = get_mean_std(1,'train')
    norm_test = get_mean_std(1,'test')
    for key,value in norm_train.items():
        norm_train = value
    for key,value in norm_test.items():
        norm_test = value
    global transform_train
    normalize_train = transforms.Normalize(**norm_train)
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((224*5,224*5)),
        # transforms.RandomCrop(32,4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_train
    ])
    train_dataset = ImageFolder(os.path.join(DATA_DIR,'train'),transform_train)
    
    normalize_test = transforms.Normalize(**norm_test)
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((224*5,224*5)),
        transforms.ToTensor(),
        normalize_test
    ])
    test_dataset = ImageFolder(os.path.join(DATA_DIR,'test'),transform_test)
    trainloader = DataLoader(
        train_dataset,batch_size=batch_size,shuffle=True,num_workers=4
    )
    testloader = DataLoader(
        test_dataset,batch_size=batch_size,shuffle=False,num_workers=4
    )
    num_classes = 2
    return trainloader, testloader, num_classes


# def show(x):
#     '''
#     show how to transform the image in first layer
#     ''' 
#     conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=3, stride=35, padding=1, bias=False)
#     bn1 = nn.BatchNorm2d(32)
#     out = F.relu(bn1(conv1(x)))
#     out = transforms.ToPILImage()(out).convert('RGB')
#     out.show()
#     pass

# transform_train = transforms.Compose([
#     transforms.CenterCrop((224*5,224*5)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
#     ])


# img_PIL = Image.open(image_path).convert('RGB')
# img_PIL.show()

# img_PIL_Tensor = transform_train(img_PIL)
# show(img_PIL_Tensor)
