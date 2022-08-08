from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import wandb

class WandbLogger():

    def __init__(self) -> None:
        pass

    def start(self, project):
        self.run = wandb.init(project=project)

    def log_cfg(self, cfg):
        wandb.config.update(cfg)

    def log(self, thing, step):
        wandb.log(thing, step=step)

    def log_metric(self, name, thing):
        wandb.log({name: thing})

    def log_image(self, thing):
        wandb.log({k: wandb.Image(v) for k, v in thing.items()})

    def log_dataset_ref(self, name, ref):
        artifact = wandb.Artifact(name, type="dataset")
        artifact.add_reference("file://" + ref)
        self.run.use_artifact(artifact)

    def log_model_weights(self, name, weights):
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(weights)
        self.run.log_artifact(artifact)

    def log_confusion_matrix(self, key, target, pred, label):
        cm = wandb.plot.confusion_matrix(y_true=target.numpy(), preds=pred.numpy(), class_names=label)
        wandb.log({key: cm})


class CometLogger():
    def __init__(self) -> None:
        pass

    def start(self, project):
        import comet_ml as comet
        self.experiment = comet.Experiment(project_name=project)

    def log_cfg(self, cfg):
        self.experiment.log_parameters(cfg)

    def log(self, thing, step):
        self.experiment.log_metrics(thing, step=step)

    def log_metric(self, name, thing):
        self.experiment.log_metrics({name: thing})

    def log_image(self, thing):
        for k, v in thing.items():
            self.experiment.log_image(v, name=k)

    def log_dataset_ref(self, name, ref):
        artifact = comet.Artifact(name, artifact_type="dataset")
        artifact.add_remote("file://" + ref)
        self.experiment.log_artifact(artifact)

    def log_model_weights(self, name, weights):
        artifact = comet.Artifact(name, artifact_type="model")
        artifact.add(weights)
        self.experiment.log_artifact(artifact)

    def log_confusion_matrix(self, key, target, pred, label):
        self.experiment.log_confusion_matrix(y_true=target, y_predicted=pred, labels=label, title=key)

import neptune.new as neptune
from neptune.new.types import File

class NeptuneLogger():
    def __init__(self) -> None:
        pass

    def start(self, project):
        self.project = "justinpinkney/" + project
        self.run = neptune.init(project=self.project)

    def log_cfg(self, cfg):
        self.run["model/parameters"] = (cfg)

    def log(self, thing, step):
        for k, v in thing.items():
            self.run[k].log(v)
        self.run["train/step"].log(step)

    def log_metric(self, name, thing):
        self.run[name].log(thing)

    def log_image(self, thing):
        for k, v in thing.items():
            self.run[k] = File.as_image(v.cpu().permute(1,2,0))

    def log_dataset_ref(self, name, ref):
        self.run[f"datasets/{name}"].track_files(ref)

    def log_model_weights(self, name, weights):
        key = "MODEL"
        # Maybe there is a better way
        try:
            self.model = neptune.init_model(name=name, key=key, project=self.project)
        except:
            self.model = neptune.init_model(model="CIF-MODEL", project=self.project)
        self.model_version = neptune.init_model_version(model="CIF-MODEL", project=self.project)

        self.model_version["model/binary"].upload(weights)
        self.model_version["run/url"] = self.run.get_url()

    def log_confusion_matrix(self, key, target, pred, label):
        # Not built in confusion matrix, but we can log any plot
        cm = ConfusionMatrixDisplay.from_predictions(y_true=target.numpy(), y_pred=pred.numpy(), display_labels=label)
        self.run[key].upload(neptune.types.File.as_html(cm.figure_))

from sklearn.metrics import ConfusionMatrixDisplay
import mlflow

class MLflowLogger():
    def __init__(self) -> None:
        mlflow.set_tracking_uri("")

    def start(self, project):
        self.experiment = mlflow.set_experiment(project)
        self.run = mlflow.start_run()

    def log_cfg(self, cfg):
        mlflow.log_params(cfg)

    def log(self, thing, step):
        mlflow.log_metrics(thing, step=step)

    def log_metric(self, name, thing):
        mlflow.log_metric(name, thing)

    def log_image(self, thing):
        for k, v in thing.items():
            mlflow.log_image(v.cpu().permute(1,2,0).numpy(), artifact_file=k+".jpg")

    def log_dataset_ref(self, name, ref):
        # can't log by reference
        mlflow.log_artifact(ref, name)

    def log_model_weights(self, name, weights):
        # Not using mlflow model registry here as that is quite different
        # mlflow models store both the the model as well as weights
        mlflow.log_artifact(weights, name)

    def log_confusion_matrix(self, key, target, pred, label):
        # Not built in confusion matrix, but we can log any plot
        cm = ConfusionMatrixDisplay.from_predictions(y_true=target.numpy(), y_pred=pred.numpy(), display_labels=label)
        mlflow.log_figure(cm.figure_, artifact_file=key+".jpg")

log_wrappers = {
    "wandb": WandbLogger,
    "comet": CometLogger,
    "neptune": NeptuneLogger,
    "mlflow": MLflowLogger,
}

SUPPORTED_LOGGERS = list(log_wrappers.keys())
def get_logger(log_type):
    assert log_type in log_wrappers, f"Don't recognise the logger {log_type}"
    return log_wrappers[log_type]()