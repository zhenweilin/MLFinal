import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.nn import Softshrink

class SGD_SCAD(Optimizer):
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
        super(SGD_SCAD,self).__init__(params,defaults)
    def __getattr__(self,key):
        print('SGD_SCAD no the attribute:{}'.format(key))
        return None
        
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
                d_p = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['timestep'] = 0
                state['timestep'] += 1

                xi = self.calculate_xi(p,theta,a=a)
                
                s = self.calculate_d(lr = lr,
                                    lambda_ = lambda_,
                                    grad = d_p,
                                    xi = xi,
                                    p = p.data,
                                    theta = theta,
                                    a = a)
                p.add_(s,alpha = 1)
                # p.add_(d_p,alpha = -lr)

        return loss

    def calculate_xi(self,p,theta,a):
        '''
        lambda calculate in calculate_d
        '''
        # vec = p.mul(-2*theta**2)
        # sca = 2*theta/(a+1)
        # bou = a/theta

        # xi = torch.zeros_like(p)
        # one = torch.ones_like(p)
        # pos_shrink = one.mul(sca)
        # pos_shrink_mid = one.mul(sca).add(-(vec.add(2*a*theta).mul(1/(a**2-1))))
        # neg_shrink_mid = one.mul(-sca).add(-(vec.add(-2*a*theta).mul(1/(a**2-1))))
        # neg_shrink = one.mul(-sca)
        # # initial this code has some problems about -one< some typo
        # pos_shrink_idx = (torch.abs(p)>=bou)&(p>0)
        # pos_shrink_mid_idx = (torch.abs(p)>1/theta)&(torch.abs(p)<bou)&(p>0)
        # neg_shrink_mid_idx = (torch.abs(p)>1/theta)&(torch.abs(p)<bou)&(p<0)
        # neg_shrink_idx = (torch.abs(p)>=bou)&(p<0)

        # xi[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        # xi[pos_shrink_mid_idx] = pos_shrink_mid[pos_shrink_mid_idx]
        # xi[neg_shrink_mid_idx] = neg_shrink_mid[neg_shrink_mid_idx]
        # xi[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        m1 = Softshrink(lambd = 1/theta)
        xi = m1(p).mul(2*theta**2/(a**2-1))
        pos = torch.sign(p).mul(2*theta/(a+1))
        pos_idx = (torch.abs(p)>a/theta)
        xi[pos_idx] = pos[pos_idx]
        return xi

    def calculate_d(self,lr,lambda_,grad,xi,p,theta,a):
        # trial_x = torch.zeros_like(p)
        sca = 2*theta/(a+1)
        # pos_shrink = p - lr*(grad - lambda_*xi + lambda_*sca)
        # neg_shrink = p - lr*(grad - lambda_*xi - lambda_*sca)
        # pos_shrink_idx = (pos_shrink > 0)
        # neg_shrink_idx = (neg_shrink < 0)
        # trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        # trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        # d = trial_x - p
        y = p.add(grad.add(xi.mul(lambda_),alpha = -1),alpha = -lr)
        m = Softshrink(lambd = lr*lambda_*sca)
        return m(y).add(p,alpha = -1)

