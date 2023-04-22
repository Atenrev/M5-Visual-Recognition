import os
import annoy
import torch
import warnings

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class Annoyer:
    # High performance approaximate nearest neighbors - agnostic wrapper
    # Find implementation and documentation on https://github.com/spotify/annoy

    def __init__(self, embedder, dataloader, emb_size=None, distance='angular', experiment_name='resnet_base', out_dir='output/', device='cuda') -> None:
        assert not (emb_size is None) and isinstance(emb_size, int),\
            f'When using Annoyer KNN emb_size must be an int. Set as None for common interface. Found: {type(emb_size)}'

        self.embedder = embedder

        # FIXME: Dataloader assumes 1 - Batch Size
        self.dataloader = dataloader
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

        self.idx_dataset2annoyer = {}
        idx_annoyer = 0
        for idx, batch in enumerate(pbar := tqdm(self.dataloader, desc='Building KNN (Annoyer)....', leave=False)):
            batch_size = len(batch[0])
            assert batch_size == 1, f'Annoyer KNN only supports batch_size = 1. Found: {batch_size}'
            anchors, positives, _ = batch

            if type(anchors[0]) == str:  # Text2Image
                positives = positives.to(self.device)
                with torch.no_grad():
                    embed = self.embedder(positives).cpu().numpy()
                embed = embed.squeeze()
                self.trees.add_item(idx, embed)
            else:  # Image2Text
                # get all captions associated with the image
                _, captions = super(type(self.dataloader.dataset), self.dataloader.dataset).__getitem__(idx)
                # store the mapping from annoyer idx to dataset idx, and index the captions
                self.idx_dataset2annoyer[idx] = []
                for i in range(len(captions)):
                    positives = captions[i]
                    positives = self.embedder.tokenizer_encode_text(positives).to(self.device)
                    print("positives.shape: ", positives.shape)
                    embed = self.embedder(positives.input_ids, positives.attention_mask).cpu().numpy()
                    embed = embed.squeeze()
                    self.trees.add_item(idx_annoyer, embed)
                    self.idx_dataset2annoyer[idx].append(idx_annoyer)
                    idx_annoyer += 1

        self.trees.build(50)
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
