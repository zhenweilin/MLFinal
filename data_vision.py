import torch
from torchvision import transforms
from PIL import Image
import os
import json,codecs
import sys
import torch.nn as nn
from DataProcess.datasets import get_mean_std
from run import selectModel
import numpy as np
import matplotlib.pyplot as plt

# image_path = os.path.join(sys.path[0],'DataProcess/datasets/train/virus/person80_virus_150.jpeg')
image_path = os.path.join(sys.path[0],'DataProcess/datasets/train/NORMAL/IM-0117-0001.jpeg')
## initial photo
img_PIL = Image.open(image_path)
# img_PIL.show()


## first step data process
norm_train = get_mean_std(1,'train')
for key ,value in norm_train.items():
    norm_train = value
def get_image_info(img_PIL):
    img_PIL = Image.open(image_path)
    normalize_train = transforms.Normalize(**norm_train)
    transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop((192*5,192*5)),
            # transforms.RandomCrop(32,4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_train
        ])
    img_PIL_Tensor = transform_train(img_PIL)
    image_info = img_PIL_Tensor.unsqueeze(0)
    return image_info

# img_PIL_Tensor = get_image_info(img_PIL)
# img = transforms.ToPILImage()(img_PIL_Tensor)
# img.show()


def get_k_layer_feature_map(model_layer,k,x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):
            x = layer(x)
            if k == index:
                return x

def show_feature_map(feature_map):
    #feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)#压缩成torch.Size([64, 55, 55])
    feature_map_num = feature_map.shape[0]#返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))#8
    plt.figure()
    for index in range(1, feature_map_num + 1):#通过遍历的方式，将64个通道的tensor拿出
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        # scipy.misc.imsave( 'feature_map_save//'+str(index) + ".png", feature_map[index - 1])
    plt.show()


if __name__ ==  '__main__':

    # 定义提取第几层的feature map
    k = 0
    image_info = get_image_info(image_path)

    model_path = './checkpoints/SGD_l1_mobilenetv1_chestX_10_edi_2.3_theta_10_lam_4.0e-04.pt'
    checkpoint = torch.load(model_path,map_location='cpu')
    model = selectModel('mobilenetv1')
    model.load_state_dict = checkpoint['model_state_dict']

    model_layer= list(model.children())
    # print(model_layer[0])
    # model_layer=model_layer[0]#这里选择model的第一个Sequential()

    feature_map = get_k_layer_feature_map(model_layer, k, image_info)
    show_feature_map(feature_map)