from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG
import pytorch_lightning as pl
from torch.optim.rmsprop import RMSprop

from dataloader import custom_data
from pytorch_lightning import loggers

import os

import cv2
import numpy as np

import timeit
from models.FNN import FingerNeuralNet
from dataloader import custom_data


def test(iterator):
    print("Starting test")
    # iouStats = Statistics()
    # fpsStats = Statistics()

    model.freeze()
    model.eval()
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # outwrite = cv2.VideoWriter('output.mp4',fourcc, 60.0, (500,500))
    its = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            its += 1
            dec_horiz, inp_horiz, dec_vert, inp_vert = model(batch)
            # print(inp_horiz.shape)
            dec_horiz = dec_horiz.detach().cpu().numpy()
            inp_horiz = inp_horiz.detach().cpu().numpy()
            dec_vert = dec_vert.detach().cpu().numpy()
            inp_vert = inp_vert.detach().cpu().numpy()
            # horizontal visualisation
            a = np.concatenate(tuple([ inp_horiz[0][:,i] for i in range(4)]), axis = -1)
            a = np.concatenate(tuple([ a[i] for i in range(4)]), axis = -1)
            cv2.imwrite('a.jpg', (a*255).astype('uint8'))

            a = np.concatenate(tuple([ dec_horiz[0][:,i] for i in range(4)]), axis = -1)
            a = np.concatenate(tuple([ a[i] for i in range(4)]), axis = -1)
            cv2.imwrite('a1.jpg', (a*255).astype('uint8'))

            # Vertical visualisation
            a = np.concatenate(tuple([ inp_vert[0][:,i] for i in range(4)]), axis = -1)
            a = np.concatenate(tuple([ a[i] for i in range(4)]), axis = -1)
            cv2.imwrite('b.jpg', (a*255).astype('uint8'))

            a = np.concatenate(tuple([ dec_vert[0][:,i] for i in range(4)]), axis = -1)
            a = np.concatenate(tuple([ a[i] for i in range(4)]), axis = -1)
            cv2.imwrite('b1.jpg', (a*255).astype('uint8'))

            exit()


    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FingerNeuralNet.load_from_checkpoint("lightning_logs/version_14/checkpoints/epoch=1-step=267.ckpt")
    model.cuda()
    dataset = custom_data()
    dataset.setup()
    test(dataset.test_dataloader())