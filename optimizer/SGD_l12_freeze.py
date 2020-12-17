import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.autograd import Variable
from torch.nn import Softshrink

class SGD_l12(Optimizer):
    def __init__(self,params,lr = 0.1, lambda_ = 1e-4,epochSize=required,Np = required,Np2 = 'inf',**kwargs):
        '''
        Np is the first stage algorithm
        Np2 is the second stage algorithm
        fix zero_element, so Np_epoch should be the same with Np_epoch+1
        '''
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid value lambda_:{}'.format(lambda_))
        if Np is not required and not 0.0 <= Np:
            raise ValueError('Invalid Np :{}'.format(Np))
        if epochSize is not required and not 0.0 < epochSize:
            raise ValueError('Invalid epochSize :{}'.format(epochSize))
        
        self.Np = Np #Np+1才开始第二阶段
        self.Np2 = Np2
        self.epochSize = epochSize
        self.iter = 0
        self.step_count = 0 #为了和外面的计数方法一致
        self.p_count = 0
        self.flag = False
        self.zero_ele = []
        # self.wei_num = 0
        # self.num_non_zero = 0
        defaults = dict(lr = lr,lambda_ = lambda_)
        super(SGD_l12,self).__init__(params,defaults)

    @torch.no_grad()
    def step(self,closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        Np2 = float('inf') if self.Np2 == 'inf' else self.Np2
        if self.step_count % (self.Np + Np2) < self.Np:
            doNp = True
            if self.iter == 0:
                print('First stage algorithm')
        else:
            doNp = False
            if self.iter == 0:
                print('Second stage algorithm')
        for group in self.param_groups:
            lr = group['lr']
            lambda_ = group['lambda_']

            for p in group['params']:
                # self.num_non_zero += len([p_[p_ != 0].numel() for p_ in p])
                if p.grad is None:
                    continue
                grad = p.grad
                # state = self.state[p]
                # self.wei_num += len([p_.numel() for p_ in p])
                # print(self.wei_num)
                if doNp:
                    inip_shape = p.shape
                    if len(p.shape) > 1:
                        s = self.calculate_d(lr,lambda_,grad,p.data)
                        p.add_(s,alpha = 1)
                    else:
                        p.add_(grad,alpha = -lr)
                    if self.step_count == self.Np - 1 and self.iter == self.epochSize-1:
                        # print('detach')
                        # self.nonzero_ele = (p.data!=0)
                        # p[self.zero_ele] = Variable(p[self.zero_ele],requires_grad = False)
                        # p[self.zero_ele].detach_()
                        p = torch.tensor(p[p != 0],requires_grad = True)
                        if p.shape != inip_shape:
                            print('shape is change')
                        if (self.step_count+1)%(self.Np+Np2) == self.Np:
                            print('transform stage')


                else:
                    if len(p.shape) > 1:
                        s = self.calculate_d2(lr,grad,p.data)
                        # p.add_(s,alpha = 1)
                        # if self.flag:
                        #     # p[self.zero_ele].backward(retain_graph = False)
                        #     p[self.zero_ele] = Variable(p[self.zero_ele],requires_grad = False)
                        #     # p[self.zero_ele].detach_()
                           
                        #     self.flag = False
                        #     self.p_count += 1
                        # self.wei_num += len(p.shape)
                        p.add_(s,alpha = 1)
                    else:
                       p.add_(grad,alpha = -lr)
                if self.p_count >= len(group['params'])-1:
                    self.p_count = 0

        if self.iter == 0:
            self.wei_num = 0
            self.num_non_zero = 0    
        self.iter += 1        
        if self.iter >= self.epochSize:
            self.step_count += 1 # 多少个epoch
            self.iter = 0 # 一个epoch里计算了多少个样本
        return loss

    
    def calculate_d(self,lr,lambda_,grad,p):
        y = p.add(grad,alpha = -lr)
        m = Softshrink(lambd = lr*lambda_)
        return m(y).add(p,alpha = -1)

    def calculate_d2(self,lr,grad,x):
        trial_x = torch.zeros_like(x)
        # if self.step_count == self.Np:
        #     '''
        #     在进行第6阶段结束后这个才被加1变成6，所以这个地方仍然用step_count == Np
        #     '''
        #     self.zero_ele = (x==0)
        #     self.flag = True
            # self.nonzero = (x != 0)
            # trial_x[self.nonzero] = -lr * grad[self.nonzero]
            # return trial_x

        trial_x = - lr * grad
        return trial_x



#换个角度，如果在第一部分算法结束后立即detach会怎么样呢，不让他经过一次backward？