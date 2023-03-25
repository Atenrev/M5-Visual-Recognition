def register_out_of_context_dataset(cfg):
    cfg.DATASETS.TRAIN = ("out_of_context_train",) 
    cfg.DATASETS.TEST = ("out_of_context_test",) 
    pass