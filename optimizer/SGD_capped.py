import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.nn import Softshrink

class SGD_capped(Optimizer):
    def __init__(self,params,lr = 1e-4, lambda_ = 1,theta = 10,**kwargs):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid value lambda_:{}'.format(lambda_))
        if not 0.0 <= theta:
            raise ValueError('Invalid value theta:{}'.format(theta))
        defaults = dict(lr = lr,lambda_ = lambda_,theta = theta)
        super(SGD_capped,self).__init__(params,defaults)

    @torch.no_grad()
    def step(self,closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lambda_ = group['lambda_']
            theta = group['theta']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['timestep'] = 0
                state['timestep'] += 1

                xi = self.calculate_xi(p,theta,lambda_)
                
                s = self.calculate_d(lr = lr,
                                    lambda_ = lambda_,
                                    grad = d_p,
                                    xi = xi,
                                    p = p.data,
                                    theta = theta)
                p.add_(s,alpha = 1)

        return loss

    def calculate_xi(self,p,theta,lambda_):
        '''
        lambda calculate in calculate_d
        '''
        # xi = torch.zeros_like(p)
        # one = torch.ones_like(p)
        # pos_shrink = one.mul(theta)
        # neg_shrink = one.mul(-theta)
        # # initial this code has some problems about -one< some typo
        # pos_shrink_idx = (torch.abs(p).mul(theta)-one>1)&(p>0)
        # neg_shrink_idx = (torch.abs(p).mul(theta)-one>1)&(p<0)    
        # xi[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        # xi[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        zero = torch.zeros_like(p)
        pos = torch.sign(p).mul(theta)
        # pos_idx = (torch.abs(p.mul(theta))>1)
        # pos_idx = (torch.abs(p.mul(theta)).add(torch.ones_like(p),alpha = -1))
        m = Softshrink(lambd = 1)
        y = m(torch.abs(p.mul(theta)))
        # xi[pos_idx] = pos[pos_idx]
        y = torch.addcmul(input = zero, tensor1 = y, tensor2 = pos)
        return y.mul(lambda_)

    def calculate_d(self,lr,lambda_,grad,xi,p,theta):
        # trial_x = torch.zeros_like(p)
        # pos_shrink = p - lr*(grad - xi + lambda_*theta)
        # neg_shrink = p - lr*(grad - xi - lambda_*theta)
        # pos_shrink_idx = (pos_shrink > 0)
        # neg_shrink_idx = (neg_shrink < 0)
        # trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        # trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        # d = trial_x - p
        y = p.add(grad.add(xi,alpha = -1),alpha = -lr)
        m = Softshrink(lambd = lambda_*theta*lr)
        d = m(y).add(p,alpha = -1)
        return d
