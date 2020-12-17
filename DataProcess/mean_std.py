import torch
import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
import json
import sys
# os.chdir(sys.path[0])
# print(sys.path[0])
# os.chdir(os.path.join(sys.path[0],'DataProcess'))
# sys.path.append(os.path.join(sys.path[0],'DataProcess'))

'''
get mean and standard devition about the datasets before train
'''
DATA_DIR = os.path.join(sys.path[0],'DataProcess/datasets')

class mean_std():
    def __init__(self,n):
        self.n = n
        self.dirs = ['train','test']
        self.means = [0]*n
        self.stdevs = [0]*n
        self.transform = transforms.Compose([
            transforms.CenterCrop((224*5,224*5)),
            transforms.ToTensor(),
            ])
        #assume you calculate multi-folder 
        self.datasets = {
            x:ImageFolder(os.path.join(DATA_DIR,x),self.transform) for x in self.dirs
        }       

    def get_mean_std(self,type = 'train'):
        '''
        type: which datasets you want to calculate like train, test
        return a file named mean_std
        '''
        num_imgs = len(self.datasets[type])
        for data in self.datasets[type]:
            img = data[0]
            for i in range(self.n):
                self.means[i] += img[i,:,:].mean()
                self.stdevs[i] += img[i,:,:].std()
        
        self.means = (np.asarray(self.means) / num_imgs).tolist()
        self.stdevs = (np.asarray(self.stdevs) / num_imgs).tolist()
        self.saving = {
            'mean':self.means,
            'std': self.stdevs
        }
        with open(os.path.join(DATA_DIR,'{}.json'.format(type)),'w') as f:
            print(self.saving)
            b = json.dumps(self.saving)
            f.write(b)
            print('get mean and std successfully {}'.format(type))

# mean_std().get_mean_std()