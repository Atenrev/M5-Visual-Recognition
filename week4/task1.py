from vision.models import Resnet, VGG19
from knn.annoyers import Annoyer, SKNNWrapper
from datautils.datasets import ZippedDataloader, MITSplitDataset
from datautils.datautils import create_mit_dataloader

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

train, test, val = create_mit_dataloader(1, '../datasets/MIT_split/', 'cuda', ProxyConfig(), inference = False)
print(train, test, val) # TODO: Why no test :(

V = np.random.random(2048)

model = Resnet(resnet = '101')
annoy = Annoyer(model, train, emb_size = 2048) # Works better with smaller emb_sizes però que li farem
annoy.fit()
print(annoy.retrieve_by_vector(V))