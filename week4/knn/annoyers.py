import annoy
import torch

from tqdm import tqdm

class Annoyer:
    # High performance approaximate nearest neighbors - agnostic wrapper
    # Find implementation and documentation on https://github.com/spotify/annoy

    def __init__(self, model, dataset, emb_size, distance = 'angular', local = 'trees.ann') -> None:
        
        self.model = model

        # FIXME: Dataloader assumes 1 - Batch Size
        self.dataloader = dataset
        self.path = local

        self.trees = annoy.AnnoyIndex(emb_size, distance)
        self.state_variables = {
            'built': False,
        }
    
    def fit(self):
        if self.state_variables['built']: raise AssertionError('Cannot fit a built Annoy')
        else: self.state_variables['built'] = True

        for idx, (image, _) in enumerate(self.dataloader):
            print(f'Building KNN... {idx} / {len(self.dataloader)}\t', end = '\r')
            with torch.no_grad():
                emb = self.model(image.float()).squeeze().cpu().numpy() # Ensure batch_size = 1

            self.trees.add_item(idx, emb)
    

        self.trees.build(10) # 10 trees
        self.trees.save(self.path)

    def load(self):
        if self.state_variables['built']: raise AssertionError('Cannot load an already built Annoy')
        else: self.state_variables['built'] = True

        self.trees.load(self.path)

    def retrieve_by_idx(self, idx, n = 50, **kwargs):
        return self.trees.get_nns_by_item(idx, n, **kwargs)
    
    def retrieve_by_vector(self, vector, n = 50, **kwargs):
        return self.trees.get_nns_by_vector(vector, n, **kwargs)

                    

