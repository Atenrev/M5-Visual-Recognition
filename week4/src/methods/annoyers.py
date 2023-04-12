import os
import annoy
import torch
import warnings

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class Annoyer:
    # High performance approaximate nearest neighbors - agnostic wrapper
    # Find implementation and documentation on https://github.com/spotify/annoy

    def __init__(self, model, dataset, emb_size=None, distance='angular', experiment_name='resnet_base', out_dir='output/', device='cuda') -> None:
        assert not (emb_size is None) and isinstance(emb_size, int),\
            f'When using Annoyer KNN emb_size must be an int. Set as None for common interface. Found: {type(emb_size)}'

        self.model = model

        # FIXME: Dataloader assumes 1 - Batch Size
        self.dataloader = dataset
        self.device = device

        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(
            out_dir, f'KNNannoy_{experiment_name}_embdize_{emb_size}_dist_{distance}.ann')

        self.trees = annoy.AnnoyIndex(emb_size, distance)
        self.state_variables = {
            'built': False,
        }

    def fit(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot fit a built Annoy')
        else:
            self.state_variables['built'] = True

        for idx, (image, _) in enumerate(self.dataloader):
            print(
                f'Building KNN... {idx} / {len(self.dataloader)}\t', end='\r')

            with torch.no_grad():
                emb = self.model(image.float().to(self.device)).squeeze(
                ).cpu().numpy()  # Ensure batch_size = 1

            self.trees.add_item(idx, emb)

        self.trees.build(10)  # 10 trees
        self.trees.save(self.path)

    def load(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot load an already built Annoy')
        else:
            self.state_variables['built'] = True

        self.trees.load(self.path)

    def retrieve_by_idx(self, idx, n=50, **kwargs):
        return self.trees.get_nns_by_item(idx, n, **kwargs)

    def retrieve_by_vector(self, vector, n=50, **kwargs):
        return self.trees.get_nns_by_vector(vector, n, **kwargs)


class SKNNWrapper:

    # Common interface for the annoyer KNN
    def __init__(self, model, dataset, distance='cosine', k=5, device='cuda', **kwargs) -> None:
        self.model = model

        # FIXME: Dataloader assumes 1 - Batch Size
        self.device = device
        self.dataloader = dataset
        self.trees = NearestNeighbors(n_neighbors=k, metric=distance)
        self.state_variables = {
            'built': False,
        }

    def fit(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot fit a built KNN')
        else:
            self.state_variables['built'] = True

        X = list()
        for idx, (image, _) in enumerate(self.dataloader):
            print(
                f'Building KNN... {idx} / {len(self.dataloader)}\t', end='\r')
            with torch.no_grad():
                emb = self.model(image.float().to(self.device)).squeeze(
                ).cpu().numpy()  # Ensure batch_size = 1

            X.append(emb)
        self.trees.fit(X)

    def load(self):
        raise NotImplementedError('Load is not implemented for sklearn KNN')

    def retrieve_by_idx(self, *args, **kwargs):
        raise NotImplementedError(
            'Retrieval by ID is not implemented for sklearn KNN')

    def retrieve_by_vector(self, vector, n=None, **kwargs):
        if not (n is None):
            warnings.warn(
                'SKLearn retrieval receives the K parameter on the constructor. Ignoring N kwarg...')
        return self.trees.kneighbors([vector], **kwargs)[-1][0].tolist()
