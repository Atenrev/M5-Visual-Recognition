import os
import csv
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from cycler import cycler

import umap
from sklearn import manifold
from sklearn.decomposition import PCA

from src.metrics import *
from src.utils import return_image_full_range
from src.datasets.mit_split import create_mit_dataloader
from src.methods.annoyers import Annoyer, SKNNWrapper
from src.models.resnet import ResNet
from src.models.vgg import VGG19


LABELS = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

OUTPUT_PATH = './outputs_task_a'
EXPERIMENT_NAME = None  # TODO: set experiment name, Adria no te n'oblidis
EXPERIMENT_DIR = os.path.join(OUTPUT_PATH, EXPERIMENT_NAME)
tensorboard_folder = os.path.join(EXPERIMENT_DIR, 'tensorboard')


def generate_sprite_image(val_ds):
    old_transform = val_ds.transform
    val_ds.transform = None  # Do not apply transforms to images when saving them to sprite

    # Gather PIL images for sprite
    images_pil = []
    for img_pt, _ in val_ds:
        img_np = img_pt.numpy().transpose(1, 2, 0) * 255
        # Save PIL image for sprite
        img_pil = Image.fromarray(img_np.astype('uint8'), 'RGB').resize((100, 100))
        images_pil.append(img_pil)

    one_square_size = int(np.ceil(np.sqrt(len(val_ds))))
    master_width = 100 * one_square_size
    master_height = 100 * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0,0,0,0)  # fully transparent
    )
    for count, image in enumerate(images_pil):
        div, mod = divmod(count, one_square_size)
        h_loc = 100 * div
        w_loc = 100 * mod
        spriteimage.paste(image, (w_loc, h_loc))

    global tensorboard_folder
    spriteimage.convert('RGB').save(f'{tensorboard_folder}/sprite.jpg', transparency=0)

    val_ds.transform = old_transform


def store_embeds_labels(embeds, labels):
    global tensorboard_folder
    # store embeddings for tensorboard's projector
    with open(f'{tensorboard_folder}/feature_vecs.tsv', 'w') as fw:
        csv_writer = csv.writer(fw, delimiter='\t')
        csv_writer.writerows(embeds)
    with open(f'{tensorboard_folder}/metadata.tsv', 'w') as file:
        for label in labels:
            file.write(f'{label}\n')

    x = """embeddings {
tensor_path: "feature_vecs.tsv"
sprite {
    image_path: "sprite.jpg"
    single_image_dim: 100
    single_image_dim: 100
}
}"""
    with open(f"{tensorboard_folder}/projector_config.pbtxt","w") as f:
        f.writelines(x)


def visualize_embeddings(embeds, labels, split_name):
    tsne = manifold.TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12,
        learning_rate="auto",
        n_iter=800,
        random_state=42,
        n_jobs=-1,
    )
    umap = umap.UMAP(random_state=42)
    pca = PCA(n_components=2, svd_solver='auto', random_state=42)
    apply_projection = {
        'tsne': tsne.fit_transform,
        'umap': umap.fit_transform,
        'pca': pca.fit_transform,
    }
    for embed_type in ['tsne', 'umap', 'pca']:
        projected_embeds = apply_projection[embed_type](embeds)
        label_set = np.unique(labels)
        num_classes = len(label_set)
        fig = plt.figure(figsize=(15, 9))
        frame = plt.gca()
        frame.set_prop_cycle(
            cycler(
                "color", [plt.cm.nipy_spectral(i)
                        for i in np.linspace(0, 0.9, num_classes)]
            )
        )
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(projected_embeds[idx, 0],
                     projected_embeds[idx, 1],
                     ".", markersize=2, label=LABELS[i])
        fig.legend(loc='outside upper right', markerscale=12)
        plt.title(f"{embed_type.upper()}")
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        global EXPERIMENT_DIR
        fig.savefig(os.path.join(EXPERIMENT_DIR, 'plots_embeddings', f"{embed_type}_{split_name}.png"))
        plt.close()


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
    train, test = create_mit_dataloader(
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

    embeds = []
    for idx in tqdm(range(len(test.dataset))):

        query, label_query = test.dataset[idx]

        V = model(query.unsqueeze(0).to(device)).squeeze()
        embeds.append(V)

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

    generate_sprite_image(test.dataset)
    embeds = torch.stack(embeds)
    store_embeds_labels(embeds, test.dataset.labels)
    visualize_embeddings(embeds, test.dataset.labels, "val")


if __name__ == "__main__":
    main()
