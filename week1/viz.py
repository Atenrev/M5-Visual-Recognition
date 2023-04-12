import scikitplot as skplt
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import kornia.augmentation as K

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mit_split import MITSplitDataset
from src.models.small_squeeze_net import SmallSqueezeNetCNN

sns.set()
device = 'cuda'


class ConfigProxy:
    num_classes = 8

    def __init__(self) -> None:
        pass


#model = build_xception_model(weights = './out/model_weights/xception_20230128-221841.h5')
model = SmallSqueezeNetCNN(ConfigProxy(), device).to(device)
checkpoint = torch.load('local/squeezeenet.pt', map_location=device)
model.load_checkpoint(checkpoint['model_state_dict'])
model.eval()

transforms = [K.Resize(255), K.Normalize(
    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]), ]

test_dirs = glob.glob(os.path.join(
    'datasets/folds/MIT_small_train_1', "test/*"))
test_dirs.sort()
classes = {x.split('/')[-1]: n for n, x in enumerate(test_dirs)}
dataval = MITSplitDataset(test_dirs, device, {},
                          transform=K.AugmentationSequential(
    *transforms,
    data_keys=["input"]
))

test_dataloader = DataLoader(
    dataset=dataval,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

# make predictions on the test data
y_score = []
test_labels = []
predicted_labels = []
string_labels = sorted(list(classes.keys()), key=lambda x: classes[x])
y_labels_argmax = []

for data in tqdm(test_dataloader):
    data = data["data"]

    with torch.no_grad():
        image, label = data['image'], data['target']
        label = label.detach().cpu().numpy()
        prediction = model(image)['logits']
        prediction = torch.nn.functional.softmax(prediction).detach().cpu().numpy()
        predictions = np.argmax(prediction, axis=1)

        for l, p in zip(label, prediction):
            y_labels_argmax.append(l.item())
            test_labels.append(string_labels[l.item()])
            y_score.append(p.tolist())
            predicted_labels.append(string_labels[np.argmax(p)])

print(sum([x == y for x, y in zip(predicted_labels, test_labels)]) / len(test_labels))
#print(len(y_score), len(predicted_labels), len(test_labels))
y_score = np.array(y_score)
skplt.metrics.plot_roc(y_labels_argmax, y_score)

plt.legend(string_labels + ['micro - average', 'macro - average'])
plt.savefig('ROCK!!!.png')
plt.clf()

skplt.metrics.plot_precision_recall(y_labels_argmax, y_score)
plt.legend(string_labels + ['micro - average', 'macro - average'])
plt.savefig('UNA PR!!!.png')
plt.clf()


# Prepare data for confusion matrix
axs_dict = classes
cat = len(axs_dict)
matrix = np.zeros((cat, cat))
for gt, pred in zip(test_labels, predicted_labels):
    matrix[axs_dict[gt], axs_dict[pred]] += 1

matrixrel = np.zeros((cat, cat))
for x in test_labels:
    for y in predicted_labels:
        matrixrel[axs_dict[x], axs_dict[y]] = round(
            100 * matrix[axs_dict[x], axs_dict[y]] / matrix[axs_dict[x], ].sum())

# Plot confusion matrix

cmap = sns.cubehelix_palette(start=1.6, light=0.8, as_cmap=True,)
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
ax = axs[0]

sns.heatmap(matrix.astype(int), annot=True, cmap=cmap, ax=ax, cbar=False)
ax.set_ylabel('GT')
ax.set_xlabel("Predicted")


ax.set_title("Absolute count")
# Rotates X-Axis Ticks by 45-degrees
ax.set_xticks(list(range(len(axs_dict))), rotation=45)
# Rotates X-Axis Ticks by 45-degrees
ax.set_yticks(list(range(len(axs_dict))), rotation=45)


ax.set_yticklabels(axs_dict, rotation=20)
ax.set_xticklabels(axs_dict, rotation=45)


sns.heatmap(matrixrel, annot=True, cmap=cmap, ax=axs[1], cbar=False, )
ax = axs[1]
ax.set_title("Relative count (%)")
ax.set_yticklabels([], rotation=0)
ax.set_xticklabels(axs_dict, rotation=45)
ax.set_xlabel("Predicted")

fig.suptitle('Confusion matrix for test set predictions')
plt.savefig('MATRIX.png')
