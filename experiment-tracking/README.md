# Experiment tracking

This example repo allows you to compare some common experiment tracking libraries:

- wandb
- mlflow
- comet
- neptune

First make sure you have setup access to the required tracking libraries if using their online services, e.g. have created an account for wandb, comet, or neptune and got your api key/credentials. For mlflow you can either log locally (assumed) or use a remote tracking server that you will need to set up.

Credentials can be passed using the appropriate environment variables. You can also populate a `.env` file with all the required variables based on the `.env.template` if you wish.

## Setup

First set up an environment, we recommend using a [Lambda GPU cloud](https://lambdalabs.com/service/gpu-cloud) instance for local testing, or for instructions running on a Run:ai cluster see below

Start a Cloud instance then:
```
git clone https://github.com/LambdaLabsML/examples.git
cd examples/experiment-tracking
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

## Viewing logs

For the various logging services (wandb, neptune, comet) go the the web page of that service and log in with the account you provided, you should then see all the logged runs. For mlflow runs are logged locally, in order to see these you need to start the mlflow ui, by ensuring that you have the mlflow python packaged installed (`pip install mlflow`) and start the ui in the same working directory as you launched the runs (`mlflow ui`).

When using Run:ai you can start a mflow ui service and then use portforwarding to access this locally:

```
./runai submit mlflowui -i python  --working-dir /workspace -v /mnt/nfs/examples/experiment-tracking/:/workspace -g 0  -- bash -c "pip install mlflow && mlflow ui"
kubectl port-forward mlflowui-0-0 5000:5000
```

Then got to `localhost:5000` in your web browser. Here `mlflowui-0-0` is the pod name, if yours is different you can check using `kubectl get pods`.
