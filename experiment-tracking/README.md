# Experiment tracking

This example repo allows you to compare some common experiment tracking libraries:

- wandb
- mlflow
- comet
- neptune

First make sure you have setup access to the required tracking libraries if using their online services, e.g. have created an account for wandb, comet, or neptune and got your api key/credentials. For mlflow you can either log locally (assumed) or use a remote tracking server that you will need to set up.

Credentials can be passed using the appropriate environment variables. You can also populate a `.env` file with all the required variables based on the `.env.template` if you wish.

## Setup

First set up an environment:

```
git clone https://github.com/LambdaLabsML/examples.git
cd examples/experiment-tracking
python -m venv .venv --prompt exp-tracking
source .venv/bin/activate
pip install -r requirements.txt
```

## Run experiments

To run a single experiment from the command line run:

```
python train.py mlflow --lr 0.1 --batch-size 64
```

pick from wandb, mlflow, comet, or neptune, as loggers.

Or to run a set of different hyper-parameters use the `run_many.sh` script:

```
./run_many.sh wandb
```

## Using Run:ai

You can also run a hyper-parameter sweep using a run:ai cluster. The following instructions assume that your nodes have access to an nfs at `/mnt/nfs/` where a copy of the examples repo is cloned, please adapt according to your own setup.

To launch the sweep do:

```
runai submit exp-tracking -i nvcr.io/nvidia/pytorch:22.03-py3 -g 0.5 --large-shm --working-dir /workspace -v /mnt/nfs/examples/experiment-tracking:/workspace --parallelism 2 --completions 8 --command -- ./run_ai_hpo.sh wandb
```

Then monitor the completion of runs using `runai logs exp-tracking --follow`. After these are complete you should be able to visit the web UI of your tracking tool to see the results.