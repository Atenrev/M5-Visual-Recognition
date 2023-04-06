from src.metrics import *
from src.utils import return_image_full_range
from src.datasets.mit_split import create_mit_dataloader
from src.methods.annoyers import Annoyer, SKNNWrapper
from src.models.resnet import ResNet
from src.models.vgg import VGG19

from tqdm import tqdm
class ProxyConfig:
    input_resize = 512

    def __init__(self) -> None:
        pass

    def __iter__(self):
        self.state = None
        return self

    def __next__(self):
        if self.state is None:
            raise StopIteration
        return 0


def main(k = 50):
    device = 'cuda'
    train, val, test = create_mit_dataloader(
        1, '../datasets/MIT_split/', ProxyConfig(), inference=False)

    model = ResNet(resnet='101').to(device)
    # Works better with smaller emb_sizes però que li farem
    annoy = Annoyer(model, train, emb_size=2048, device=device, distance='angular')
    try: annoy.load()
    except:
        annoy.state_variables['built'] = False
        annoy.fit()

    mavep = []
    mavep25 = []
    top_1_acc = []
    top_5_acc = []
    top_10_acc = []

    for idx in tqdm(range(len(test.dataset))):

        query, label_query = test.dataset[idx]

        V = model(query.unsqueeze(0).to(device)).squeeze()

        query = (return_image_full_range(query))
        nns, distances = annoy.retrieve_by_vector(V, n=k, include_distances = True)
        labels, images = list(), list()
        

        for nn in nns:
            _, label = train.dataset[nn]
            labels.append(int(label == label_query))

        mavep.append(calculate_mean_average_precision(labels, distances))
        mavep25.append(calculate_mean_average_precision(labels[:26], distances[:26]))
        top_1_acc.append(calculate_top_k_accuracy(labels, k = 1))
        top_5_acc.append(calculate_top_k_accuracy(labels, k = 5))
        top_10_acc.append(calculate_top_k_accuracy(labels, k = 10))
    
    print(
        "Metrics: ",\
        f"\n\tmAveP@50: {np.mean(mavep)}", \
        f"\n\tmAveP@25: {np.mean(mavep25)}", \
        f"\n\ttop_1 - precision: {np.mean(top_1_acc)}",\
        f"\n\ttop_5 - precision: {np.mean(top_5_acc)}",\
        f"\n\ttop_10 - precision: {np.mean(top_10_acc)}"
    )




if __name__ == "__main__":
    main()
