import argparse
import csv
import os
import time
import sys
os.chdir(sys.path[0])
# print(os.getcwd())
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from DataProcess import Datasets
from torch import optim

from utils import get_logger
from thop import profile, clever_format
from torch.autograd import Variable



from model.mobilenetv1 import MobileNet
from model.resnet import ResNet18, ResNet50,ResNet101,ResNet152
import optimizer
from utils import check_accuracy, compute_F, compute_value

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate','-lr',default=0.1,type = float
    )
    parser.add_argument(
        '--model',choices=['mobilenetv1','resnet18','resnet50'],type = str, required = True
    )
    # parser.add_argument(
    #     '--dataset_name',choices = ['cifar10','fashion_mnist'],type = str,required = True
    # )
    parser.add_argument(
        '--optimizer',choices=['pDCAe_exp','SGD_l1','SGD_capped','SGD_SCAD','SGD_mcp','pDCAe_nobeta','SGD_l12','SGD_l12_freeze']
    )
    parser.add_argument(
        '--batch_size',default=128,type = int
    )
    parser.add_argument(
        '--theta',default = 10,type = int,help = 'the greater theta, the better fit to norm zero'
    )
    # 这个地方theta 的取值为什么要取10？
    parser.add_argument(
        '--max_epoch',default=200,type = int
    )
    parser.add_argument(
        '--lambda_',default = 0.0001,type = float
    )
    parser.add_argument(
        '--a',default = 2,type = float,help = 'This parameter is using in SCAD peanlty'
    )
    parser.add_argument(
        '--edition',default = 1.0,type = float, help = 'This is code edition'
    )
    parser.add_argument(
        '--Np',default = 100, type = int, help = 'This is how many epoches in first stage'
    )
    parser.add_argument(
        '--Np2',default = 100, type = int, help  = 'This is how many epoches in second stage'
    )
    return parser.parse_args()

def selectModel(modelName):
    if modelName == 'resnet18':
        print('modelName: ResNet18')
        model = ResNet18()
    elif modelName == 'mobilenetv1':
        print('modelName: MobileNetV1')
        model = MobileNet()
    elif modelName == 'resnet50':
        print('modelName: ResNet50')
        model = ResNet50()
    elif modelName == 'resnet101':
        print('modelName:ResNet101')
        model = ResNet101()
    elif modelName == 'resnet152':
        print('modelName:ResNet152')
        model = ResNet152()
    else:
        raise ValueError('Invalid model name:{}'.format(modelName))
    return model





def main():
    args = parseArgs()
    lr = args.learning_rate
    batch_size = args.batch_size
    modelName = args.model
    opt = args.optimizer
    theta = args.theta
    max_epoch = args.max_epoch
    lambda_ = args.lambda_
    a = args.a
    edition = args.edition
    Np = args.Np
    Np2 = args.Np2
    dataset_name = 'chestX'




    #Manually set parameters for testing
    # lr = 0.1
    # dataset_name = 'chestX'
    # batch_size = 128
    # modelName = 'mobilenetv1'
    # opt = 'SGD_l12'
    # theta = 10
    # lambda_ = 0.00014
    # max_epoch = 300
    # a = 2
    '''
    13是detach版本与　14是requires_grad为False的版本
    0.000001 是　detach 版本
    '''

    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainloader, testloader, num_classes = Datasets(batch_size)

    model = selectModel(modelName)
    model = model.to(device=device)
    weights = [w for name, w in model.named_parameters() if "weight" in name] 
    num_features = sum([w.numel() for w in weights])
    num_samples = len(trainloader)*batch_size
    criterion = nn.CrossEntropyLoss()
    optimize = optimizer.__dict__[opt](**{
        'params' : model.parameters(),
        'lr':lr,
        'theta':theta,
        'lambda_':lambda_,
        'epochSize':len(trainloader),
        'Np':100,
        'Np2':100,
    })
    if opt !='rda':
        scheduler = StepLR(optimize, step_size = 30, gamma = 0.1)
    os.makedirs('./results_{}'.format(opt),exist_ok=True)
    if a == 2:
        setting = '{}_{}_{}_{}_edi_{}_theta_{}_lam_{:.1e}'.format(opt,modelName,dataset_name,theta,edition,theta,lambda_)
    else:
        setting = '{}_{}_{}_{}_edi_{}_theta_{}_lam_{:.1e}_a{}'.format(opt,modelName,dataset_name,theta,edition,theta,lambda_,a)
    if opt == 'SGD_l12_freeze':
        setting = '{}_{}_{}_{}_edi_{}_theta_{}_lam_{:.1e}_a{}_Np_{}'.format(opt,modelName,dataset_name,theta,edition,theta,lambda_,a,Np)
    csvname = os.path.join('./results_{}'.format(opt),setting+'.csv')
    print('Result are saving to the CSV file {}'.format(csvname))
    

    alg_start_time = time.time()
    epoch = 1
    if os.path.exists(os.path.join('./checkpoints',setting+'.pt')):
        checkpoint = torch.load(os.path.join('./checkpoints',setting+'.pt'),map_location = torch.device('cpu'))
        epoch += checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimize.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        csvfile = open(csvname,'a',newline='')
        fieldnames = ['epoch','F_value','f_value','penaltyvalue','density','train_time','accuracy','stage']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames,delimiter = ',')
        print('using the previous model')
    else:
        csvfile = open(csvname,'w',newline='')
        fieldnames = ['epoch','F_value','f_value','penaltyvalue','density','train_time','accuracy','stage']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames,delimiter = ',')
        writer.writeheader()

    logger = get_logger('log_{}_{}'.format(modelName,opt))
    # flops_input = torch.randn(1,3,32,32).cuda()# the image size
    # flops, params = profile(model, inputs = (flops_input, ))
    # flops, params = clever_format([flops,params],'%.3f')
    # logger.info(
    #     f'model:{modelName},flops:{flops},params:{params}'
    # )





    while True:
        epoch_start_time = time.time()
        if epoch > max_epoch:
            break
        for _,(X,y) in enumerate(trainloader):
            # iterate every minibatch
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)
            f1 = criterion(y_pred, y)
            optimize.zero_grad()#zero grad before every iteration 
            f1.backward()
            optimize.step()
        # logger.info(model.parameters())

        if opt != 'rda':
            scheduler.step()

        train_time = time.time() - epoch_start_time

        F,f,penaltyvalue = compute_value(
            trainloader,model,weights,criterion,theta,lambda_,penalty=opt,a=a
        )
        
        density = sum([torch.sum(w != 0).item() for w in weights])/num_features
        # density = optimize.num_non_zero/optimize.wei_num
        accuracy = check_accuracy(model,testloader)
        print('epoch:{}'.format(epoch),'time:{}'.format(train_time))
        print('density:{}'.format(density))
        print("F:{},\n f:{},\n penaltyvalue:{},\n accuracy:{}".format(F,f,penaltyvalue,accuracy))
        writer.writerow({
            'epoch':epoch,
            'F_value':F,
            'f_value':f,
            'density':density,
            'train_time':train_time,
            'penaltyvalue':penaltyvalue,
            'accuracy':accuracy,
            'stage':optimize.stage
        })
        csvfile.flush()
        if epoch >= max_epoch:
            torch.save({
                'epoch':epoch,
                'accuracy':accuracy,
                'lr':scheduler.get_last_lr()[0],
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimize.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
            },os.path.join('checkpoints',setting+'.pt'))
            print('save model successfully')
        epoch += 1
 

    alg_time = time.time()-alg_start_time
    writer.writerow({'train_time':alg_time/epoch})
    os.makedirs('checkpoints',exist_ok=True)
    csvfile.close()

if __name__ == '__main__':
    main()