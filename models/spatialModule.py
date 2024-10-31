from torch import nn
# import pytorch_spiking
import torch
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from models.config import FNNconfigs

'''
The entire data can be represented using:

1. Type of curve.
2. Max amplitude.
3. x,y value.
4. Velocity of palpation.
'''

class VAEencBlock(nn.Module):
    def __init__(self, LD):
        super(VAEencBlock, self).__init__()
        FNNconfigs.__init__(self,)
        # 512x32

        self.latentdim = LD

        encoderModules = []
        hiddenDims = [4, 32, 128, 128, 64, 32]
        inChannels = 4
        for hDim in hiddenDims:
            encoderModules.append(
                nn.Sequential(
                    nn.Conv2d(inChannels, inChannels, kernel_size= 3, stride= 2, padding  = 1, groups = inChannels),
                    nn.Conv2d(inChannels, hDim, kernel_size= 1),
                    nn.BatchNorm2d(hDim),
                    nn.LeakyReLU()
                )
            )
            inChannels = hDim
        self.encoder = nn.Sequential(*encoderModules)
        self.fc1 = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.fc21 = nn.Linear(64,self.latentdim)
        self.fc22 = nn.Linear(64,self.latentdim)

    def forward(self, x):
        b,c,w,h = x.shape
        a = self.encoder(x)
        # print(a.shape)
        a = self.fc1(a)
        mean = self.fc21(a)
        var = self.fc22(a)

        return mean, var

class VAEdecBlock(nn.Module):
    def __init__(self, LD):
        super(VAEdecBlock, self).__init__()
        FNNconfigs.__init__(self,)

        hiddenDims = [4, 32, 128,64, 16]

        self.latentdim = LD

        hiddenDims.reverse()
        decoderModules = []
        for i in range(len(hiddenDims) - 1):
            decoderModules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddenDims[i], hiddenDims[i + 1], kernel_size=3, stride = 2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hiddenDims[i + 1]),
                    nn.LeakyReLU())
            )

        decoderModules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddenDims[-1], hiddenDims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hiddenDims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hiddenDims[-1], out_channels= 4, kernel_size= 3, stride = 1,  padding = 1),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*decoderModules)

        self.fc = nn.Sequential(
            nn.Linear(self.latentdim, 64), 
            nn.ReLU(),
            nn.Linear(64, 256), 
            nn.ReLU()
        )


    def forward(self, x):

        features = self.fc(x).view(-1,16,16,1)
        a = self.decoder(features)

        return a

def reparametrize(mu, logvar):
    std = torch.exp(logvar * 0.5)
    eps = torch.rand_like(std)
    ret = eps * std + mu
    return ret