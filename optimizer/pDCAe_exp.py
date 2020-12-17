import torch 
from torch.optim.optimizer import Optimizer, required
import math
from torch.nn import Softshrink
#这个地方softshrink里面应该是用Ｃ写的

class pDCAe(Optimizer):
    def __init__(self, params, lr = 1e-4,theta=10,lambda_ = 0.000001,epochSize = 200,**kwargs):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= theta:
            raise ValueError('Invalid approximation coefficient theta :{}'.format(theta))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid value lambda_:{}'.format(lambda_))
        self.epochSize = epochSize
        self.iter = 0
        defaults = dict(lr = lr,theta = theta,lambda_ = lambda_)
        super(pDCAe,self).__init__(params,defaults)

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
                grad = p.grad

                state = self.state[p]
                #Initialize the k
                if len(state) == 0:
                    state['k_'] = state['k'] = 1
                    state['x_'] = torch.clone(p).detach()

                k_,k = state['k_'],state['k']
                # calculate beta
                beta = (k_-1)/k
                k_ = k
                k = (1+math.sqrt(1+4*k**2))/2
                # calculate xi
                xi = self.calculate_xi(state['x_'],theta,lambda_)

                '''
                next is about how to calculate new p which is from a segment function,
                but there is a problem about how to calculate gradient in y_t
                '''

                d = self.calculatex_d(lr,p,grad,xi,theta,lambda_)
                p.add_(d)
                x = torch.clone(p)
                p = x.mul(1+beta).add(state['x_'],alpha = -beta)
                state['x_'] = torch.clone(x).detach()
                if self.iter >= self.epochSize:
                    self.iter = 0
                    p = state['x_']

        self.iter += 1
        

    def calculate_xi(self,p,theta,lambda_):
        one = torch.ones_like(p)
        zero = torch.zeros_like(p)
        xi2 = one.add(torch.exp(torch.abs(p).mul(-theta)),alpha = -1)
        xi1 = torch.sign(p).mul(lambda_*theta)
        xi = torch.addcmul(input = zero,tensor1 = xi1, tensor2= xi2)
        return xi

    def calculatex_d(self,lr,p,grad,xi,theta,lambda_):
        trial_x = torch.zeros_like(p)
        pos_shrink = p - lr*(grad - xi + lambda_*theta)
        neg_shrink = p - lr*(grad - xi - lambda_*theta)
        pos_shrink_idx = (pos_shrink > 0)
        neg_shrink_idx = (neg_shrink < 0)
        trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        trial_x = trial_x - p
        return trial_x






