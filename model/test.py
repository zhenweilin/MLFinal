import torch.nn as nn
import torch.nn.functional as functional

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes,out_channels=planes,kernel_size = 3, stride =stride, padding = 1,bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels = planes, out_channels = planes,kernel_size=3,padding=1,bias=False,stride = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = functional.relu(out)
        return out


a = BasicBlock(64,32)
# print([k for k in a.named_parameters()])
print([w.numel() for name, w in a.named_parameters() if "weight" in name])