import torch
import os
import logging
from logging.handlers import TimedRotatingFileHandler

def compute_F(trainloader,model,weights,criterion,lmbda):
    f = 0.0
    device = next(model.parameters()).device
    # covert the next device
    for index, (X,y) in enumerate(trainloader):
        X = X.to(device)
        y = y.to(device)
        y_pred = model.forward(X)
        f1 = criterion(y_pred,y)
        f += float(f1)
    f /= len(trainloader)
    norm_l1_x_list = []
    #here weights use item means storge 
    for w in weights:
        norm_l1_x_list.append(torch.norm(w,1).item())
    norm_l1_x = sum(norm_l1_x_list)#sum of norm1 
    F = f + lmbda * norm_l1_x
    # F is considered penalty's objective value
    return F, f, norm_l1_x


def check_accuracy(model,testloader):
    correct = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for (X,y) in testloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            _,predicted = torch.max(y_pred.data,1)
            total += y.size(0)#so y contains a minibatch
            correct += (predicted == y).sum().item()

    model = model.train() # set model to train pattern
    accuracy = correct / total
    return accuracy


def compute_value(trainloader,model,weights,criterion,theta,lambda_,penalty = 'SGD_l1',a = 2):
    f = 0.0
    device = next(model.parameters()).device
    # covert the next device
    for index, (X,y) in enumerate(trainloader):
        X = X.to(device)
        y = y.to(device)
        y_pred = model.forward(X)
        f1 = criterion(y_pred,y)
        f += float(f1)
    f /= len(trainloader)
    penalty_list = []
    for w in weights:
        one = torch.ones_like(w)
        if penalty == 'pDCAe_nobeta':
            penalty_w = torch.ones_like(w).add(torch.exp(torch.abs(w).mul(-theta)),alpha = -1)
        elif penalty == 'pDCAe_exp':
            penalty_w = torch.ones_like(w).add(torch.exp(torch.abs(w).mul(-theta)),alpha = -1)
        elif penalty == 'SGD_l1':
            penalty_w = w
        elif penalty == 'SGD_capped':
            penalty_w = calculate_capped(w = w,theta = theta)
        elif penalty == 'SGD_SCAD':
            penalty_w = calculate_SCAD(theta = theta,a = a,w = w)
        elif penalty == 'SGD_mcp':
            penalty_w = calculate_mcp(theta=theta,a = a, w = w)
        elif penalty == 'SGD_l12':
            penalty_w = w
        elif penalty == 'SGD_l12_freeze':
            penalty_w = w

        penalty_list.append(torch.norm(penalty_w,1).item())

    penaltyvalue = sum(penalty_list)
    F = f + penaltyvalue*lambda_
    #calculate lambda here
    return F,f,penaltyvalue*lambda_
   


def calculate_capped(w,theta):
    one = torch.ones_like(w)
    idx = (theta*torch.abs(w)<1)
    one[idx] = theta*torch.abs(w)[idx]
    return one 

def calculate_SCAD(theta,a,w):
    one = torch.ones_like(w)
    value1 = torch.abs(w).mul(2*theta/(a+1))
    value2 = torch.abs(w).mul(2*a*theta).add(w.pow(2).mul(-theta**2)).add(one,alpha = -1).mul(-1/(a**2-1))
    idx1 = (torch.abs(w)<=1/theta)
    idx2 = (torch.abs(w)>=1/theta)&(torch.abs(w)<=a/theta)
    one[idx1] = value1[idx1]
    one[idx2] = value2[idx2]
    return one

def calculate_mcp(theta,a,w):
    pos = torch.ones_like(w).mul(a*theta**2)
    neg_shrink = torch.abs(w).mul(2*theta).add(w.pow(2).mul(-1/a**2))
    neg_shrink_idx = (torch.abs(w) <= a*theta)
    pos[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
    return pos

def get_logger(name, log_dir='./log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    error_handler = TimedRotatingFileHandler(error_name,
                                             when='D',
                                             encoding='utf-8')
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger



class AverageMeter(object):
    def __init__(self,name,fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.store = []
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.store = []
    def update(self,val,n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

        # 类的一些静态函数，类函数，普通函数，全局变量以及一些内置的属性都是放在__dict__里面了，
        # 存储了一些self.**的东西
        # __str__对于使用没有什么用，但是方便我们直接对这个东西进行打印返回一些东西
        # 利用fmt, 使其在打印的时候能够标准化输出
        # 对其使用　str 命令时也是输出这个命令
    def save(self,val):
        self.store.append(val)

    def __str__(self):
        fmtstr = '{name}{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self,num_batches, meters, prefix = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self,batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self,num_batches):
        num_digits = len(str(num_batches//1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
        # 前面这个fmt 里面有一个可以填的位置，用于填第几个batch[len(test_loader)d　表示
        # 的意思应该是00001 这样进行表示]
        # 后面fmt 里面填的是batch中的数量

