## Lab 5 - Cross-modal Retrieval

[Presentation slides](https://docs.google.com/presentation/d/1gfIFUH8qz5ue8yad1Zp9mZdpseCHdp4RkYCZdrdq--A/edit?usp=sharing)

[Technical Report](https://www.overleaf.com/read/wczvtbgkzmtz)
.

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

- Install other dependencies and fix distutils problem
```
pip install opencv-python fiftyone setuptools==59.5.0 scikit-image
```


## Execution


Simply running `bash run_all.sh` launches training of all models for all tasks.

To execute training of each particular model, you may use the files named `task_<task>_option<option>.slurm` individually, or the python commands therein.

The retrieval with the trained models is executed with files `task_<task>_option<option>_retrieval.slurm`. Make sure to pass in the path to the pretrained weights.

You can also use a model checkpoint produced during training to generate qualitative results with `create_embeddings_plot.py`.