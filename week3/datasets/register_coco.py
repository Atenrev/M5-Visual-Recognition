def register_coco_dataset(cfg):
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ("coco_test",)
    pass