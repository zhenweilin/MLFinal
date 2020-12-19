import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.nn import Softshrink

class SGD_mcp(Optimizer):
    def __init__(self,params,lr = 1e-4, lambda_ = 1,theta = 10,a = 2,**kwargs):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid value lambda_:{}'.format(lambda_))
        if not 0.0 <= theta:
            raise ValueError('Invalid value theta:{}'.format(theta))
        if not 0.0 <= a:
            raise ValueError('Invalid value a:{}'.format(a))
        defaults = dict(lr = lr,lambda_ = lambda_,theta = theta,a = a)
        super(SGD_mcp,self).__init__(params,defaults)

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
            a = group['a']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['timestep'] = 0
                state['timestep'] += 1

                xi = self.calculate_xi(p,theta,a)
                
                s = self.calculate_d(lr = lr,
                                    lambda_ = lambda_,
                                    grad = grad,
                                    xi = xi,
                                    p = p.data,
                                    theta = theta)

                p.add_(s,alpha = 1)
                # y = p.add(grad.add(xi.mul(lambda_),alpha = -1),alpha = -lr)
                # m = Softshrink(lambd = lambda_*theta*lr)
                # p = m(y)
        return loss

    def calculate_xi(self,p,theta,a):
        '''
        lambda calculate in calculate_d
        '''
        # one = torch.ones_like(p)
        # pos = one.mul(2*theta)
        # mid = p.mul(2/theta)
        # neg_shrink = one.mul(-2*theta)
        # # initial this code has some problems about -one< some typo
        # mid_idx = (torch.abs(p).add(one.mul(a*theta),alpha = -1)<=0)
        # neg_shrink_idx = (p.add(one.mul(a*theta))<0)

        # pos[mid_idx] = mid[mid_idx]
        # pos[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        # return pos
        less = p.mul(2/a)
        pos =  torch.abs(p)
        pos_idx = (torch.abs(p)>a*theta)
        less[pos_idx] = torch.sign(p).mul(2*theta)[pos_idx]
        return less

    def calculate_d(self,lr,lambda_,grad,xi,p,theta):
        # trial_x = torch.zeros_like(p)
        # pos_shrink = p - lr*(grad - lambda_*xi + lambda_*theta)
        # neg_shrink = p - lr*(grad - lambda_*xi - lambda_*theta)
        # pos_shrink_idx = (pos_shrink > 0)
        # neg_shrink_idx = (neg_shrink < 0)
        # trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        # trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        # d = trial_x - p
        # return d
        y = p.add(grad.add(xi.mul(lambda_),alpha = -1),alpha = -lr)
        m = Softshrink(lambd = lambda_*theta*lr)
        return m(y).add(p,alpha = -1)
        
