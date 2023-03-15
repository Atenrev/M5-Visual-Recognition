## Lab 2

[Presentation slides](https://docs.google.com/presentation/d/1C0G-nqxH_7CE-lY5INbEw5qNCo7IHPPgs7JmohEAivA/edit?usp=sharing)


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