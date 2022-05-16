import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from resnet18 import resnet18

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,8,kernel_size=3,padding=0),
                        nn.BatchNorm2d(8, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(8,8,kernel_size=3,padding=0),
                        nn.BatchNorm2d(8, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(8,8,kernel_size=3,padding=1),
                        nn.BatchNorm2d(8, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(3))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(8,8,kernel_size=3,padding=1),
                        nn.BatchNorm2d(8, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(3))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),-1)
        return out

class FullyConnect(nn.Module):
    #539
    def __init__(self):
        super(FullyConnect, self).__init__()
        self.fc1 = nn.Linear(10240, 2048)
        self.fc2 = nn.Linear(2048, 16*8*(2*5+2)) # 40*40一个boudbox
        # 16 * 10 * (2 * 5 + 1) = 1760

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        return out

if __name__ == '__main__':
    input = torch.randn(1,3,224,224)

    # cnn = CNNEncoder()
    fc = FullyConnect()
    # out = cnn(input)
    # print(out.shape)
    # out1 = fc(out)
    # print(out1.shape)

    resnet18_model = resnet18(True)
    out = resnet18_model(input)
    print(out.shape)

    out1 = fc(out)
    print(out1.shape)