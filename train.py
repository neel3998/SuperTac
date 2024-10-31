import pytorch_lightning as pl
from models.FNN import FingerNeuralNet
from dataloader import custom_data

from pytorch_lightning.callbacks import ModelCheckpoint

# dataset = MNIST('../mnist_data',  download=True, train=True, transform=transforms.Compose([ transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                                                            # transforms.Normalize((0.1307,), (0.3081,))]))
dataset = custom_data()
dataset.setup()

# train_loader = DataLoader(dataset)
model = FingerNeuralNet()
# model = FingerNeuralNet.load_from_checkpoint("lightning_logs/version_26099/checkpoints/epoch=6-step=1189.ckpt")


trainer = pl.Trainer(gpus = 1, max_epochs = 120)
trainer.fit(model, dataset)

# JUGAD - Using a gpu means you have to change the .cuda() thing in pytorch-spiking/functional.py----> SpikingActivation--forward line 66 and add .cuda()