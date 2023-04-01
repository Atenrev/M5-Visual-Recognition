from vision.models import Resnet
from knn.annoyers import Annoyer
from datautils.datasets import ZippedDataloader

from torch.utils.data import DataLoader
# Shit file for unitary testing

model = Resnet(resnet = '101')
dataset = ZippedDataloader('/home/adri/Pictures/zipper.zip')
dataloader = DataLoader(dataset, batch_size = 1)

annoy = Annoyer(model, dataloader, emb_size = 2048) # Works better with smaller emb_sizes però que li farem
annoy.fit()
print(annoy.retrieve_by_idx(3))