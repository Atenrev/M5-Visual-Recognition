import argparse
import json
import glob
import os
import cv2
import numpy as np

from itertools import groupby
from skimage import measure
from tqdm import tqdm
from datetime import datetime
from pycocotools import mask as maskUtils


CAR_ID = 1
PEDESTRIAN_ID = 2


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for creating COCO annotations for KITTI MOTS.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--ds_dir", "-m", type=str, default="/home/mcv/datasets/KITTI-MOTS/",
                        help="Dataset dir")
    parser.add_argument("--split", "-s", type=str, default="training",
                        help="Split (training, testing)")

    return parser.parse_args()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def close_contour(contour):
    """
    Original code: https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
    """
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    """
    Original code: https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')

    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """
    Converts a binary mask to COCO polygon representation
    Original code: https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def create_split(image_paths, split: str = "training"):
    print(f"Creating {split} split...")
    now = datetime.now()
    now_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info = {
        "year": now.year,
        "version": 1.0,
        "description": "KITTI MOTS annotations in COCO format.",
        "contributor": "Paul Voigtlaender et al.",
        "url": "https://www.vision.rwth-aachen.de/page/mots",
        "date_created": now_formatted,
    }

    categories = [
        {
            "supercategory": None,
            "id": CAR_ID,
            "name": "car",
        },
        {
            "supercategory": None,
            "id": PEDESTRIAN_ID,
            "name": "pedestrian",
        },
    ]

    images = []
    annotations = []
    total_pedestrians = 0
    total_cars = 0
    next_ann_id = 1

    for img_path in tqdm(image_paths):
        # Image info creation
        image_arr = cv2.imread(img_path)
        height, width = image_arr.shape[:2]

        img_relative_path = os.path.relpath(img_path, os.path.join(args.ds_dir, args.split))
        image_id = int(img_relative_path.split("/")[-2]) * 1000000 + int(img_relative_path.split("/")[-1].split(".")[0]) + 1
        image = {
            "id": image_id,
            "file_name": img_relative_path,
            "height": height,
            "width": width,
            "license": None,
        }

        images.append(image)    

        # Annotations creation
        # Reference code: https://github.com/KevinJia1212/MOTS_Tools/blob/master/mots2coco.py
        ann_mask_path = os.path.join(args.ds_dir, "instances", img_relative_path.replace("image_02/", ""))
        annotation = cv2.imread(ann_mask_path, -1)

        h, w = annotation.shape[:2]
        ids = np.unique(annotation)
        is_crowd = 1 if "crowd" in img_relative_path else 0

        for id in ids:
            if id in [0, 10000]: # ignore background
                continue
            else:
                class_id = id // 1000
                if class_id == CAR_ID:
                    total_cars += 1
                elif class_id == PEDESTRIAN_ID:
                    total_pedestrians += 1
                else:
                    continue

            instance_mask = np.zeros((h, w, ),dtype=np.uint8)
            mask = annotation == id
            instance_mask[mask] = 255

            # Convert to COCO format
            # Reference code: https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
            binary_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            binary_mask_encoded = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

            area = maskUtils.area(binary_mask_encoded)
            if area < 1:
                print(f"Warning: skipping {'car' if class_id == CAR_ID else 'pedestrian'} instance with no segmentation")
                continue

            bounding_box = maskUtils.toBbox(binary_mask_encoded)

            if is_crowd == 1:
                segmentation = binary_mask_to_rle(binary_mask)
            else:
                segmentation = binary_mask_to_polygon(binary_mask, tolerance=2)

                if not segmentation:
                    print(f"Warning: skipping {'car' if class_id == CAR_ID else 'pedestrian'} instance with no segmentation")
                    continue
            
            annotations.append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "iscrowd": is_crowd,
                "area": area,
                "bbox": bounding_box,
                "segmentation": segmentation,
                "width": binary_mask.shape[1],
                "height": binary_mask.shape[0],
            })
            next_ann_id += 1

    coco_dict = {
        "info": info,
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(f"labels_{split}.json", "w") as f:
        json.dump(coco_dict, f, cls=NpEncoder)

    print(f"Created {split} split with {len(images)} images, {total_pedestrians} pedestrians and {total_cars} cars.")


def main(args: argparse.Namespace):
    print("Creating COCO annotations")

    # Dataset format: /[training,testing]/image_02/0001/000131.png
    all_image_paths = sorted(glob.glob(os.path.join(args.ds_dir, f"{args.split}/image_02/*/*.png")))
    # Sequences 2, 6, 7, 8, 10, 13, 14, 16 and 18 were chosen for the validation set, the remaining sequences for the training set.
    # Training image paths 
    training_image_paths = [path for path in all_image_paths if int(path.split("/")[-2]) not in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    # Testing image paths 
    testing_image_paths = [path for path in all_image_paths if int(path.split("/")[-2]) in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    create_split(training_image_paths, split="training")
    create_split(testing_image_paths, split="testing")
    

if __name__ == "__main__":
    args = _parse_args()
    main(args)