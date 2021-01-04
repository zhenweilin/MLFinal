import torch
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets
import sys
import os
import json,codecs
import numpy as np
from mean_std import mean_std
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


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
    test_batch_size = 234 + 390
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
        transforms.CenterCrop((192*5,192*5)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_train
    ])
    train_dataset = ImageFolder(os.path.join(DATA_DIR,'train'),transform_train)
    
    normalize_test = transforms.Normalize(**norm_test)
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((192*5,192*5)),
        transforms.ToTensor(),
        normalize_test
    ])
    #192*5 32 160*2 25
    #CenterCrop -> Resize
    test_dataset = ImageFolder(os.path.join(DATA_DIR,'test'),transform_test)
    trainloader = DataLoader(
        train_dataset,batch_size=batch_size,shuffle=True,num_workers=4
    )
    testloader = DataLoader(
        test_dataset,batch_size=test_batch_size,shuffle=False,num_workers=4
    )
    num_classes = 3
    return trainloader, testloader, num_classes

def Datasets_val(batch_size = 8+8+9):
    norm_test = get_mean_std(1,'test')
    for key,value in norm_test.items():
        norm_test = value
    normalize_test = transforms.Normalize(**norm_test)
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((192*5,192*5)),
        transforms.ToTensor(),
        normalize_test
    ])
    val_dataset = ImageFolder(os.path.join(DATA_DIR,'val'),transform_val)
    valloader = DataLoader(
        val_dataset,batch_size=batch_size,shuffle=True,num_workers=4
    )
    return valloader
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
