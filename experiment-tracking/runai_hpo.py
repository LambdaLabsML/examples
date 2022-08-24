import runai.hpo
import os
from train import main
import typer

grid = {
    "batch_size": [128, 256, 512],
    "lr": [0.1, 0.01, 0.001],
}

def run_hpo(
    log_type:str=typer.Argument("wandb"),
    project:str="cifar-hpo",
):
    hpo_root = '/workspace/hpo'
    if not os.path.exists(hpo_root):
        os.mkdir(hpo_root)
    hpo_experiment = '%s_%s' % (os.getenv('JOB_NAME'), os.getenv('JOB_UUID'))
    runai.hpo.init(hpo_root, hpo_experiment)
    strategy = runai.hpo.Strategy.GridSearch
    config = runai.hpo.pick(grid=grid, strategy=strategy)

    main(log_type=log_type, project=project, **config)

if __name__ == "__main__":
    typer.run(run_hpo)
