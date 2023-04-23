## Lab 2 - Introduction to Object detection and Instance Segmentation with Detectron2

[Presentation slides](https://docs.google.com/presentation/d/1C0G-nqxH_7CE-lY5INbEw5qNCo7IHPPgs7JmohEAivA/edit?usp=sharing)

[Overleaf document](https://www.overleaf.com/read/wczvtbgkzmtz)


## Set up
- Create a conda environment with Python 3.9
```
conda create -n m5w2 python=3.9
conda activate m5w2
```

- Install [Pytorch 1.10.1 cuda 11.3](https://pytorch.org/get-started/previous-versions/)
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

- Install Detectron2
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

- Install other dependencies and fix distutils problem
```
pip install opencv-python fiftyone setuptools==59.5.0 scikit-image
```


## Execution

To execute inference and finetuning of Mask RCNN and Faster RCNN, simply run `python main.py` with the appropiate arguments:

```
Week2 - Pretrained and finetuned detectors.

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE, -m MODE  Mode (train, eval, draw_seg, draw_sequence)
  --seed SEED, -s SEED  Seed
  --model MODEL, -mo MODEL
                        Model (mask_rcnn, faster_rcnn)
  --checkpoint CHECKPOINT, -ch CHECKPOINT
                        Model weights path
  --head_num_classes HEAD_NUM_CLASSES, -hnc HEAD_NUM_CLASSES
                        Number of classes for the head. If not set, uses the
                        default model head.
  --map_kitti_to_coco   Map KITTI classes to COCO classes
  --dataset_dir DATASET_DIR, -tr DATASET_DIR
                        Train dataset name
  --labels_dir LABELS_DIR, -ld LABELS_DIR
                        Train dataset name
  --dry
  --resume_or_load
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size
  --epochs EPOCHS, -ep EPOCHS
                        Training epochs
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Training epochs
  --num_gpus NUM_GPUS, -ng NUM_GPUS
                        Number of GPUs
  --output_dir OUTPUT_DIR, -od OUTPUT_DIR
                        Output directory
  --sequence SEQUENCE, -seq SEQUENCE
                        Sequence to draw in draw_sequence mode
```