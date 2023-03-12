# M5-Visual-Recognition


## Group 1

- Alex Carrillo Alza ([email](mailto:21alexth@gmail.com))

- Adria Molina Rodriguez ([email](mailto:amolina@cvc.uab.cat))

- Sergi Masip Cabezas ([email](mailto:sergi.masip@autonoma.cat))

- Alvaro Francesc Budria Fernandez ([email](mailto:alvaro.francesc.budria@estudiantat.upc.edu))


## Lab 1

[Overleaf document](https://www.overleaf.com/read/krjjggkdhpsb)

[Presentation slides](https://docs.google.com/presentation/d/1N0aDFoihjSk5I_r0FaBP8MKEkNiQIScsjYzAy7u0WtA/edit?usp=sharing)


## Get started 

### Training and evaluation

To train the model, run the following command:

```sh
python main.py
  --mode "train"
  --model               Model to run
  --dataset_config      Dataset config to use
  --trainer_config      Trainer params to use
  --dataset_dir         Dataset directory path
  --load_checkpoint     Path to model checkpoint
  --batch_size          Batch size
  --seed                Seed to use
```

To evaluate the model, change the ```--mode``` to "eval".


### Configuration

Every model, dataset, and trainer is configured in a configuration file. The configuration file is a YAML file. The configuration files are located in the ```configs``` folder. In case you want to add a new model, dataset, or trainer, you should create a new configuration file and add it to the ```configs``` folder, as well as the corresponding model or dataset script in ```src```.

#### Dataset configuration
For each dataset, you need a configuration file in the ```configs/datasets``` folder. The file must contain the "name" parameter, which is the same as the name of the dataset script in ```src/datasets``` that will be used to load the dataset.

#### Model configuration
For each model, you need a configuration file in the ```configs/models``` folder. The name of the file must be the same as the name of the model script in ```src/models``` that will be used to load the model. The file must contain the the following parameters:
``` YAML
classname: <class name of the model>

metrics:
    - <list of metrics to use>
```

### Trainer configuration
For the trainer, you need a configuration file in the ```configs/trainers``` folder. The file must contain the the following parameters:

``` YAML
epochs: <number of epochs>
project_name: <name of the project (used for trackers)>
entity: <name of the entity (used for wandb)>
runs_path: <path to the runs folder>
report_path: <path to the report folder>

optimizer:
    type: <type of optimizer>
    # ... parameters of the optimizer
```

The runs folder is where the training logs will be saved. The report folder is where the evaluation reports will be saved.


### Code structure

```sh
# Run dataset setup and feature extraction
./tools
    extract_visual_features/
    setup_dataset/

# Store dataset
./datasets
    MIT_split/
        ...
    ...

# Store configuration
./config
    datasets/
    models/
    trainers/

# Create you own models or datasets
.src/
    models/                                               <= This is where you should add your own models. They should inherit from the BaseModel class. 
        base_model.py
    datasets/                                             <= This is where you should add your own datasets. They must inherit from the ```BaseDataset``` class.
        base_dataset.py

# Run the model
./main.py
```