import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.nn import Softshrink

class SGD_l1(Optimizer):
    def __init__(self,params,lr = 1e-4, lambda_ = 1,**kwargs):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid value lambda_:{}'.format(lambda_))
        defaults = dict(lr = lr,lambda_ = lambda_)
        super(SGD_l1,self).__init__(params,defaults)

    @torch.no_grad()
    def step(self,closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lambda_ = group['lambda_']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if len(p.shape)>1:
                    s = self.calculate_d(lr,lambda_,grad,p.data)
                    p.add_(s,alpha = 1)
                else:
                    p.add_(grad,alpha = -lr)
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
