import os
import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn.manifold import TSNE


def plot_both_embeddings(image_embeddings, text_embeddings, output_path: str = "plots/both_embeddings.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Concatenate the two sets of embeddings
    embeddings = np.concatenate((image_embeddings, text_embeddings), axis=0)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2,
                           perplexity=30,
                           early_exaggeration=12,
                           learning_rate="auto",
                           n_iter=800,
                           random_state=42,
                           n_jobs=-1,
                           )
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Split the t-SNE results back into the two sets of embeddings
    embeddings1_tsne = embeddings_tsne[:len(image_embeddings)]
    embeddings2_tsne = embeddings_tsne[len(image_embeddings):]

    # Plot the two sets of embeddings using different colors and markers
    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings1_tsne[:, 0], embeddings1_tsne[:, 1], c='orange', marker='o', s=50, alpha=0.7, label='Image Embeddings')
    plt.scatter(embeddings2_tsne[:, 0], embeddings2_tsne[:, 1], c='blue', marker='s', s=50, alpha=0.7, label='Text Embeddings')
    plt.legend(fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
