from src.metrics import plot_retrieved_images
from src.utils import return_image_full_range
from src.datasets.mit_split import create_mit_dataloader
from src.methods.annoyers import Annoyer, SKNNWrapper
from src.models.resnet import ResNet
from src.models.vgg import VGG19


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


def main():
    device = 'cuda'
    train, val, test = create_mit_dataloader(
        1, './datasets/MIT_split/', ProxyConfig(), inference=False)

    model = ResNet(resnet='101').to(device)
    # Works better with smaller emb_sizes però que li farem
    annoy = Annoyer(model, train, emb_size=2048, device=device)
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


if __name__ == "__main__":
    main()
