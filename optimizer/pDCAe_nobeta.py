import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.nn import Softshrink

class pDCAe_nobeta(Optimizer):
    def __init__(self, params, lr = 1e-4, theta=10,lambda_ = 10,**kwargs):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= theta:
            raise ValueError('Invalid approximation coefficient theta :{}'.format(theta))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid value lambda_:{}'.format(lambda_))
        defaults = dict(lr = lr,theta = theta,lambda_ = lambda_)
        super(pDCAe_nobeta,self).__init__(params,defaults)

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            theta = group['theta']
            lambda_ = group['lambda_']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                #Initialize the k
                if len(state) == 0:
                    state['timestep'] = 0
                
                one = torch.ones_like(p.data,memory_format=torch.preserve_format)
                # calculate xi
                xi = self.calculate_xi(p,theta,lambda_)
            
                '''
                next is about how to calculate new p which is from a segment function,
                but there is a problem about how to calculate gradient in y_t
                '''
                d = self.calculatex_d(lr,p.data,grad,xi,theta,lambda_)
                p.add_(d)
                state['timestep']+=1
        return loss

    def calculate_xi(self,p,theta,lambda_):
        one = torch.ones_like(p)
        zero = torch.zeros_like(p)
        xi2 = one.add(torch.exp(torch.abs(p).mul(-theta)),alpha = -1)
        xi1 = torch.sign(p).mul(lambda_*theta)
        xi = torch.addcmul(input = zero,tensor1 = xi1, tensor2= xi2)
        return xi

    def calculatex_d(self,lr,p,grad,xi,theta,lambda_):
        # trial_x = torch.zeros_like(p)
        # pos_shrink = p - lr*(grad - xi + lambda_*theta)
        # neg_shrink = p - lr*(grad - xi - lambda_*theta)
        # pos_shrink_idx = (pos_shrink > 0)
        # neg_shrink_idx = (neg_shrink < 0)
        # trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        # trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        # trial_x = trial_x - p
        y = p.add(grad.add(xi,alpha = -1),alpha = -lr)
        m = Softshrink(lambda_*theta*lr)
        trial_x = m(y).add(p,alpha = -1)
        return trial_x

