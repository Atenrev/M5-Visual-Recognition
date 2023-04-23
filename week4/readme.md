## Lab 4

[Presentation slides](https://docs.google.com/presentation/d/1Nc-LMoexcwWQh2YC-LRhskNEfGTeSG66W5IjuYabq2U/edit?usp=sharing)

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

- Install other dependencies and fix distutils problem
```
pip install opencv-python fiftyone setuptools==59.5.0 scikit-image
```


## Execution

To run code for task a: `python task_a.py`.

To execute task b, you may run `bash run_task_b_experiments.sh` which launches all studied configurations, or you may execute one of the individual `.slurm` files that are called from within.

For training models in task c, run `bash run_task_c_experiments.sh`, or again, you may just execute one of the individual calls from within. 

To run retrieval on models from tasks a, b, and c, use `python run_retrieval.py`, passing in `--model_weights_path <path>`.

Similarly, `python task_e.py` and `task_e_evaluate.py` train and evaluate models for task e, respectively.

Finally, to obtain additional qualitative results and evaluation, run `python run_retrieval_qualitative.py`. You will need to specify a dataset, and a path to a set of pretrained weights.