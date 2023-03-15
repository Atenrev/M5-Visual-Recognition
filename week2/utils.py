import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def load_kitti_and_map_to_coco(dataset_name: str, dataset_dir: str, labels_path: str):
    # Load the annotations from the JSON file
    with open(labels_path, "r") as f:
        annotations = json.load(f)
    
    class_dict = {1: 2, 2: 0}
    detectron_anns = []

    # Update the image paths and add the annotations
    for image in annotations["images"]:
        image["image_id"] = image["id"]
        image["file_name"] = os.path.join(dataset_dir, image["file_name"])
        image["annotations"] = []

        for ann in annotations["annotations"]:
            if ann["image_id"] == image["id"]:
                ann["category_id"] = class_dict[ann["category_id"]]
                ann["bbox_mode"] = BoxMode.XYWH_ABS, # This is being converted to a tuple for some reason
                ann["bbox_mode"] = ann["bbox_mode"][0] # So we need to get the first element, fuck this
                image["annotations"].append(ann)

        detectron_anns.append(image)
        

    # Create the DatasetCatalog entry
    DatasetCatalog.register(
        dataset_name,
        lambda: detectron_anns,
    )

    coco_names = [""] * 81
    coco_names[0] = "person"
    coco_names[2] = "car"

    # Define the metadata
    metadata = {
        "thing_classes": coco_names,
        # "thing_dataset_id_to_contiguous_id": {1: 2, 2: 0},
    }

    # Create the MetadataCatalog entry
    MetadataCatalog.get(dataset_name).set(**metadata)