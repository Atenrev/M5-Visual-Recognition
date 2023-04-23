## Lab 3 - Elephant in the Room

[Presentation slides](https://docs.google.com/presentation/d/1fATsuFsUoD_CjUBYmI8Pr8WGOHp-kMw_m_ujcg-kkGQ/edit#slide=id.g1f9a58d00d7_0_0)

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

To execute the tasks, run `python main.py`, specifying which task is to be run:

```
Week3 - Challenges of Object Detection and Instance Segmentation.

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE, -m MODE  Mode (task_a, task_b, task_c, task_d, task_e)
  --seed SEED, -s SEED  Seed
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory
  --model MODEL, -mo MODEL
                        Model (mask_rcnn, faster_rcnn)
  --checkpoint CHECKPOINT, -ch CHECKPOINT
                        Model weights path
  --load_dataset LOAD_DATASET, -tr LOAD_DATASET
                        Load dataset
  --sequence SEQUENCE, -seq SEQUENCE
                        Sequence to draw in draw_sequence mode
```
