import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.nn import Softshrink
class SGD_l12(Optimizer):
    def __init__(self,params,lr = 0.1, lambda_ = 1e-4,epochSize=required,Np = required,Np2 = 'inf',**kwargs):
        '''
        Np is the first stage algorithm
        Np2 is the second stage algorithm
        '''
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid value lambda_:{}'.format(lambda_))
        if Np is not required and not 0.0 <= Np:
            raise ValueError('Invalid Np :{}'.format(Np))
        if epochSize is not required and not 0.0 < epochSize:
            raise ValueError('Invalid epochSize :{}'.format(epochSize))
        
        self.Np = Np
        self.Np2 = Np2
        self.epochSize = epochSize
        self.iter = 0
        self.step_count = 0
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
                if p.grad is None:
                    continue
                grad = p.grad
                # state = self.state[p]
                if doNp:
                    if len(p.shape) > 1:
                        s = self.calculate_d(lr,lambda_,grad,p.data)
                        p.add_(s,alpha = 1)
                    else:
                        p.add_(grad,alpha = -lr)
                else:
                    if len(p.shape) > 1:
                        s = self.calculate_d2(lr,grad,p.data)
                        p.add_(s,alpha = 1)
                    else:
                        p.add_(grad,alpha = -lr)

        self.iter += 1
        if self.iter >= self.epochSize:
            self.step_count += 1 # 多少个epoch
            self.iter = 0 # 一个epoch里计算了多少个样本
        return loss

    
    def calculate_d(self,lr,lambda_,grad,p):
        # trial_x = torch.zeros_like(x)
        # pos_shrink = x - lr*grad - lr * lambda_
        # neg_shrink = x - lr*grad + lr * lambda_
        # pos_shrink_idx = (pos_shrink > 0)
        # neg_shrink_idx = (neg_shrink < 0)
        # trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        # trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        # d = trial_x - x
        # return d
        y = p.add(grad,alpha = -lr)
        m = Softshrink(lambd = lr*lambda_)
        return m(y).add(p,alpha = -1)

    def calculate_d2(self,lr,grad,x):
        trial_x = torch.zeros_like(x)
        nonzero = (x != 0)
        trial_x[nonzero] = grad.mul(-lr)[nonzero]
        return trial_x

