from copy import deepcopy
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

import umap
from sklearn import manifold
from sklearn.decomposition import PCA


def load_data_dict(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(file_path, map_location=torch.device('cpu'))
    if type(data) == type(dict()):
        return data  # ['embeddings'], data['targets'], data['preds'], data['att_time'], ['att_inputs']
    raise TypeError(f'Expected data container to be of type `{type(dict)}` but got `{type(data)}` instead.')


def obtain_tSNE_projection(
    embeddings: Union[torch.Tensor, np.array],
    ) -> np.array:

    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
    if len(embeddings.shape) != 2:
        raise RuntimeError(
            (f'Expected input embeddings to be two-dimensional tensor'
            f' but got a `{len(embeddings.shape)}-dimensional tensor instead.`')
        )
    tsne = manifold.TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12,
        learning_rate="auto",
        n_iter=1000,
        random_state=41,
        n_jobs=-1,
    )
    Y = tsne.fit_transform(embeddings)
    return Y


def obtain_UMAP_projection(
    embeddings: Union[torch.Tensor, np.array],
    ) -> np.array:

    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
    if len(embeddings.shape) != 2:
        raise RuntimeError(
            (f'Expected input embeddings to be two-dimensional tensor'
            f' but got a `{len(embeddings.shape)}-dimensional tensor instead.`')
        )
    Y = umap.UMAP().fit_transform(embeddings)
    return Y


def obtain_PCA_projection(
    embeddings: Union[torch.Tensor, np.array],
    ) -> np.array:

    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
    if len(embeddings.shape) != 2:
        raise RuntimeError(
            (f'Expected input embeddings to be two-dimensional tensor'
            f' but got a `{len(embeddings.shape)}-dimensional tensor instead.`')
        )
    pca = PCA(n_components=2, svd_solver='auto')
    Y = pca.fit_transform(embeddings)
    return Y


def plot_projection(
    Y: np.array,
    class_labels: Union[List[int], List[str], torch.Tensor],
    title: str,
    ) -> None:
    if type(class_labels) == torch.Tensor:
        class_labels = deepcopy(class_labels).tolist()
    # toy example
    # class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # y = [0, 1, 2, 2, 2, 2, 3, 3, 4, 4,  5, 5]

    dpi = 100
    fig, ax = plt.subplots(figsize=(850/dpi,850/dpi), dpi=dpi)

    ax.axis('off')
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=class_labels, cmap='tab08')
    # scatter = plt.scatter(x, y, c=class_labels, cmap='tab10')
    # lgd = ax.legend(handles=scatter.legend_elements()[0], labels=labels, ncol=1, bbox_to_anchor=(1.04,1))
    ax.set_title(f'{title}')
    # plt.savefig(f'./outputs/tsne_{model}_{data_type}_{split}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(f'./outputs/{title}.png', bbox_inches='tight')
    plt.close()


title = '...'
data = load_data_dict(f'./outputs/inference_{title}.pt')

targets = data['targets']# + 1
# preds = data['preds'] + 1
# logits = data['logits']
# labels = obtain_labels(data['targets'])  # obtain_labels(data['targets'])
labels = ['Open_country'] #...

Y = obtain_tSNE_projection(data['embeddings'])
print(f'Plotting projections; title = {title}...', flush=True)
plot_projection(Y, targets,)
