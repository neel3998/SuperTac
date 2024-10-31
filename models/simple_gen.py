from torch import nn
# import pytorch_spiking
import torch
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from models.config import FNNconfigs

class ConvSkipBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvSkipBlock, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride =1 , padding =1),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        x1 = self.c1(x)
        x1 = torch.cat((x,x1), dim= 1)
        return x1

class SuperResdecBlock(nn.Module):
    def __init__(self, LD):
        super(SuperResdecBlock, self).__init__()
        FNNconfigs.__init__(self,)

        hiddenDims = [1, 16,
                      32, 32]

        self.latentdim = LD

        decoderModules = []
        for i in range(0, len(hiddenDims) - 2, 2):
            decoderModules.append(
                nn.Sequential(
                    # ConvSkipBlock(hiddenDims[i], hiddenDims[i+1]),
                    nn.ConvTranspose2d(hiddenDims[i], + hiddenDims[i+1], kernel_size=3, stride = 1, padding=1),
                    nn.ConvTranspose2d(hiddenDims[i+1] , hiddenDims[i + 2], kernel_size=3, stride = 2, padding=1, output_padding=1),
                    nn.LeakyReLU())
            )

        decoderModules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddenDims[-2], hiddenDims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(hiddenDims[-1], hiddenDims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        self.decoder = nn.Sequential(*decoderModules)

        self.fc = nn.Sequential(
            # nn.Flatten(1),
            nn.Linear(self.latentdim, 4), 
            nn.ReLU(),
            nn.Linear(4, 4), 
            nn.ReLU(),

        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(hiddenDims[-1], 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, LF_vert, LF_horiz):
        SR_horiz = []

        for i in range(len(LF_horiz)):
            temp = self.fc(LF_horiz[i]).view(-1,1,4,1)
            temp = self.decoder(temp)
            SR_horiz.append(temp)

        a = torch.cat(tuple([SR_horiz[i] for i in range(4)]), dim = -1)
        a = self.conv1(a)
        # exit()
        # decoded = torch.stack(SR_temp, dim=1)
        # decoded = torch.cat(tuple([ decoded[:,i] for i in range(4)]), dim = -2)
        # # print(decoded.shape)
        # decoded = self.conv1(decoded)


        return a