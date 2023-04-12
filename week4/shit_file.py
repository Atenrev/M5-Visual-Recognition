from src.models.resnet import ResNet
from src.models.vgg import VGG19
from src.methods.annoyers import Annoyer, SKNNWrapper
from src.datasets.zipped import ZippedDataloader

from torch.utils.data import DataLoader
import numpy as np


# Shit file for unitary testing
V = np.random.random(2048)

model = ResNet(resnet = '101')
dataset = ZippedDataloader('/home/adri/Pictures/zipper.zip')
dataloader = DataLoader(dataset, batch_size = 1)

annoy = SKNNWrapper(model, dataloader, emb_size = 2048) # Works better with smaller emb_sizes però que li farem
annoy.fit()
print(annoy.retrieve_by_vector(V))


# model = VGG19()
dataset = ZippedDataloader('/home/adri/Pictures/zipper.zip')
dataloader = DataLoader(dataset, batch_size = 1)

annoy = Annoyer(model, dataloader, emb_size = 2048) # Works better with smaller emb_sizes però que li farem
annoy.fit()
print(annoy.retrieve_by_vector(V))