import os
import glob
import sys
import shutil

DATA_DIR = './datasets'
CLASS = 'PNEUMONIA'
CLASS1 = 'virus'
CLASS2 = 'bacteria'

# print(os.listdir(DATA_DIR))

for dataname in os.listdir(DATA_DIR):
    if os.path.splitext(dataname)[1] != '.json':#目录下包含.json的文件
        data = os.path.join(DATA_DIR,dataname,CLASS)
        virus = []
        bacteria = []
        for f in os.listdir(data):
            if 'virus' in f:
                virus.append(f)
            else:
                bacteria.append(f)
        os.makedirs(os.path.join(DATA_DIR,dataname,CLASS1),exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR,dataname,CLASS2),exist_ok=True)
        for v in virus:
            _v = os.path.join(data,v)
            shutil.copy(_v,os.path.join(DATA_DIR,dataname,CLASS1))
        for b in bacteria:
            _b  = os.path.join(data,b)
            shutil.copy(_b,os.path.join(DATA_DIR,dataname,CLASS2))
