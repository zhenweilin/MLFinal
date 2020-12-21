'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn #neural network
import torch.nn.functional as F

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        # in_channels, out_channels,kernel_size,stride,padding control the number of zero-padding , groups is the number of conv2
        #padding is to add extra pixels of filler around the boundary of our input image,typically, we set the values of the extra pixels to zero
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_channels = in_planes, out_channels = out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # the step 1 is 3x3 conv->batchnorm->relu
        out = F.relu(self.bn2(self.conv2(out))) # the step 2 is 1x1 conv->batchnorm->relu
        # print('**',out.shape)
        return out
 

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    # haven't seen this grammer before, put cfg here and it will become self attribute automatically
    def __init__(self, num_classes=3):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3, stride=35, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32) #set the first layer's in_planes
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes # so we can connect the net
        return nn.Sequential(*layers) #note that this need to be unzip

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    print(net)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
