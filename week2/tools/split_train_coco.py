import argparse
import json


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for splitting COCO datasets.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--labels", "-m", type=str, default="./labels_training.json",
                        help="Dataset dir")
    parser.add_argument("--val_seqs", "-s", type=list, default=["0019", "0020"],
                        help="Split (training, testing)")

    return parser.parse_args()


def main(args: argparse.Namespace):
    with open(args.labels, "r") as f:
        labels_org = json.load(f)

    labels_train = dict(labels_org)
    labels_val = dict(labels_org)

    labels_train["images"] = []
    labels_val["images"] = []

    # Image file_name format: /image_02/0001/000131.png
    # Split the dataset into train and val
    # Last 2 sequences are used for validation
    train_indices = []
    val_indices = []

    for i, im in enumerate(labels_org["images"]):
        if im["file_name"].split("/")[-2] in args.val_seqs:
            val_indices.append(i)
            labels_val["images"].append(im)
        else:
            train_indices.append(i)
            labels_train["images"].append(im)

    train_image_ids = [labels_org["images"][i]["id"] for i in train_indices]
    test_image_ids = [labels_org["images"][i]["id"] for i in val_indices]

    labels_train["annotations"] = [
        ann for ann in labels_org["annotations"]
        if ann["image_id"] in train_image_ids
    ]
    labels_val["annotations"] = [
        ann for ann in labels_org["annotations"]
        if ann["image_id"] in test_image_ids
    ]

    with open("labels_train_split.json", "w") as f:
        json.dump(labels_train, f)

    with open("labels_val_split.json", "w") as f:
        json.dump(labels_val, f)

    print(f"Created train split with {len(labels_train['images'])} images and {len(labels_train['annotations'])} annotations.")
    print(f"Created val split with {len(labels_val['images'])} images and {len(labels_val['annotations'])} annotations.")


if __name__ == "__main__":
    args = _parse_args()
    main(args)