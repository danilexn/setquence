# SetQuence
*by Daniel León-Periñán @ ZIH, TU Dresden*
*documentation in progress*

This repository contains code used for the Master thesis "Scalable Deep Set Representations for Genomics Optimizing and Applying SETQUENCE to Large-Scale Cancer Datasets", exploring architectural modifications and optimizations for large-scale training of SetQuence. Specifically, the optimizations were implemented with the [*alpha* partition of ZIH's HPC systems](https://doc.zih.tu-dresden.de/jobs_and_resources/alpha_centauri/) in mind; each of the 34 nodes in this partition consist of 8x NVIDIA A100 GPUs (40GB vRAM), 2 x AMD EPYC CPU 7352 and 1 TB RAM.

## What is SetQuence?
SetQuence is a Deep Neural Network architecture to integrate sets of genomic sequences, via Language Modeling, to perform supervised tasks. For example, SetQuence is showcased in [our publication](https://ieeexplore.ieee.org/document/9863058/) for the goal of tumor type classification from patient mutomes.

## 1. First steps
SetQuence can be installed on Unix-based or Windows systems supporting Python >=3.6. Please, make sure this requirement is met. We recommend installing SetQuence in a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/).

#### 1.1 Create and activate a new virtual environment

```
conda create -n setquence python=3.6
conda activate setquence
```

#### 1.2 Install the package and other requirements

(Required) First, install DNABERT and its dependencies. DNABERT is a core component of _SetQuence_ and _SetOmic_, as it is the encoder for biological sequence data. Installing all requirements for this package will solve most dependencies for SetQuence.

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

(Required) Analogously, proceed to install SetQuence

```
git clone https://github.com/danilexn/setquence
cd setquence
python3 -m pip install .
```

## 2. Supervised training of SetQuence models
### 2.1 Training dataset
To train a SetQuence model, you need a dataset in any of the [supported formats](). You may find an example dataset (patient mutomes from TCGA pan-cancer for tumor type classification) in [this link]().

### 2.2 Configuring the model
Then, you need to create a **json** file containing the configuration for the model. You can find a template in [configs/config_template.json](https://github.com/danilexn/setquence/blob/main/configs/config_template.json). 

This file contains 5 regions, that do not need to be in order. More detailed explanations of each setting can be found at [configs/template_instructions.md](https://github.com/danilexn/setquence/blob/main/configs/template_instructions.md). 

### 2.3 Training
I provide in [bash/train.sh](https://github.com/danilexn/setquence/blob/main/bash/train.sh) a script to run training on the tested computational settings, see above. This can be executed as:

```
bash/train.sh \
    -c [route to .config file] \
    -e [route to experiment dir] \
    -g [number of GPUS per node] \
    -n [number of nodes] \
    -a [project or account name] \
    -p [port] \
    -s [python executable location; does not require anaconda]
```

You can also write your own training scripts, for instance:
```bash
#!/bin/bash
# SLURM configuration here (see https://slurm.schedmd.com/sbatch.html); for example:
# Load modules; for example (uncomment to run)
# module load modenv/hiera 

# Script variables that can be configured
CONFIG_FILE="example_config.json" # replace by your file
EXPERIMENT_DIR="logs_example" # replace by a directory for logging

# Environment variables optionally used by SetQuence
export SLURM_MASTER_PORT=12345 # where DDP will open a port
export SETQUENCE_LOG_WANDB=1 # enabla wandb logging. Requires wandb and configuring and account
export SETQUENCE_LOG_WANDB_PROJECT="example_project"
export SETQUENCE_LOG_WANDB_ENTITY="example_entity"
export SETQUENCE_LOG_WANDB_FREQ=100 # number of steps until wandb metrics are updated
export SETQUENCE_LOG_WANDB_RUNNAME="example_training"

python -m setquence -c $CONFIG_FILE -e $EXPERIMENT_DIR
```

As well, SetQuence modules are written in a modular manner, such that training can be performed from notebooks, such as this [example](https://github.com/danilexn/setquence/blob/main/notebooks/train_example.ipynb).


## 3. Contact
You can open an [issue](https://github.com/danilexn/setquence/issues/new/choose) or contact me via [email](mailto://daniel.leon-perinan@mailbox.tu-dresden.de)