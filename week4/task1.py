from vision.models import Resnet, VGG19
from knn.annoyers import Annoyer, SKNNWrapper
from datautils.datasets import ZippedDataloader, MITSplitDataset
from datautils.datautils import create_mit_dataloader, return_image_full_range
import matplotlib.pyplot as plt
from vision.metrics import plot_retrieved_images

from torch.utils.data import DataLoader
import numpy as np

class ProxyConfig:
    input_resize = 512
    def __init__(self) -> None:
        pass

    def __iter__(self):
        self.state = None
        return self

    def __next__(self):
        if self.state is None: raise StopIteration
        return 0

device = 'cuda'
train, test, val = create_mit_dataloader(1, '../datasets/MIT_split/', 'cuda', ProxyConfig(), inference = False)


model = Resnet(resnet = '101').to(device)
annoy = Annoyer(model, train, emb_size = 2048, device = device) # Works better with smaller emb_sizes però que li farem
annoy.load()

query, label_query = test.dataset[234]

V = model(query.unsqueeze(0).to(device)).squeeze()

query = (return_image_full_range(query))
nns = annoy.retrieve_by_vector(V, n=15)

labels, images = list(), list()

for nn in nns:

    img, label = train.dataset[nn]
    images.append(return_image_full_range(img))
    labels.append(int(label == label_query))

plot_retrieved_images(query, images, labels)