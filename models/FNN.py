import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.spatialModule import reparametrize, VAEencBlock, VAEdecBlock
from models.config import FNNconfigs

from models.simple_gen import SuperResdecBlock
from models.GAN_loss import ContentLoss
from torch.autograd import Variable
from models.simple_gen import SuperResdecBlock
import torch.cuda.amp as amp
import cv2

torch.autograd.set_detect_anomaly(True)

class cLoss():
    def __init__(self,):
        self.closs_desc = ContentLoss()
    
    def exec(self,SR, HR):
        return(self.closs_desc(SR, HR))

class FingerNeuralNet(pl.LightningModule):

    def __init__(self):
        super(FingerNeuralNet, self).__init__()
        FNNconfigs.__init__(self,)

        self.learning_rate = 0.002
        # self.automatic_optimization = False

        self.latentDims = 32
        self.smallenc_vert = VAEencBlock(self.latentDims) # 10 denotes the number of latent features.
        self.smalldec_vert = VAEdecBlock(self.latentDims)

        # self.smallenc_horiz = VAEencBlock(self.latentDims) # 10 denotes the number of latent features.
        # self.smalldec_horiz = VAEdecBlock(self.latentDims)

        # self.SR_dec = SuperResdecBlock(self.latentDims)
        # self.cLoss = cLoss()
        # self.d_criterion = nn.BCEWithLogitsLoss()
    
        
    def createsensorImage(self, inp):
        a = []
        for i in range(4):
            a.append(torch.mean(inp[:,0,-1,32*i:32*(i+1)], dim = 1))
        
        a = torch.stack(a).permute(1,0)
        return a

    def forward(self, batch):
        HRimg = batch['hr'].float()
        # print(batch['z_vert'].shape)
        # _,_, decoded_vae_horiz, input_vae_horiz = self.run_VAE(batch,'horiz')
        _,_, decoded_vae_vert, input_vae_vert = self.run_VAE(batch,'horiz')

        # print(LF)
        # return decoded_vae_horiz, input_vae_horiz, decoded_vae_vert, input_vae_vert
        return decoded_vae_vert, input_vae_vert, decoded_vae_vert, input_vae_vert
        # a[-1] --> SR

    
    def run_VAE(self, batch, orientation):
        KLDlosses = 0
        latent_features = []
        decs = []
        targets = []
        BCELoss = 0
        # print(batch['v_vert'].shape)
        for i in range(4):
            inp = batch['v_'+orientation][:,i]

            mean,var = self.smallenc_vert(inp.float().cuda())
            sample = reparametrize(mean, var)
            dec = self.smalldec_vert(sample)


            decs.append(dec)
            targets.append(batch['z_'+orientation][:,i])

            kld = -0.5 * torch.mean(1 + var - torch.pow(mean, 2) - torch.exp(var)).float()

            KLDlosses += kld
            # BCELoss += F.binary_cross_entropy(dec, batch['z_'+orientation][:,i].float().cuda(), reduction='mean')

            latent_features.append(sample)

        decoded = torch.stack(decs, dim=1)
        targets = torch.stack(targets, dim=1)
        # print(decoded.shape)
        a = torch.cat(tuple([ decoded[:,:,i] for i in range(4)]), dim = -1)
        # print(a.shape)
        a = torch.cat(tuple([ a[:,i] for i in range(4)]), dim = -1)

        b = torch.cat(tuple([ targets[:,:,i] for i in range(4)]), dim = -1)
        b = torch.cat(tuple([ b[:,i] for i in range(4)]), dim = -1)

        BCELoss = F.binary_cross_entropy(a, b.float().cuda(), reduction='mean')


        loss = BCELoss
        return loss, latent_features, decoded, targets.float().cuda()
    
    def training_step(self, batch, hiddens):

        # V1_opt= self.optimizers(use_pl_optimizer=True)

        HRimg = batch['hr'].float()
        target_label = batch['class']

        # V1_opt.zero_grad()
        vae_loss_vert,LF_vert, decoded_vae, inputs_vae = self.run_VAE(batch,'horiz' )
        # self.manual_backward(vae_loss_vert)
        # V1_opt.step() 


        self.log('vae_vert', vae_loss_vert ,  on_step=True, on_epoch=False, 
                prog_bar=True, logger= True)

            
        return vae_loss_vert



    def configure_optimizers(self):

        V1_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

        return V1_optimizer #, V2_optimizer, G_optimizer