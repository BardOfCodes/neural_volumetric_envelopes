"""
Logger with W&B integration.
"""

import wandb

class WandbLogger():
    def __init__(self, project_name: str, entity: str, exp_name: str, train_config: dict):
        self.wandb = wandb
        self.wandb.init(project=project_name, entity=entity, name=exp_name, config=train_config)

    def __del__(self):
        self.finish()

    def log(self, metrics, step):
        self.wandb.log(metrics, step=step)

    def save(self):
        self.wandb.save()

    def finish(self):
        self.wandb.finish()

    def watch(self, model):
        self.wandb.watch(model)

    def log_model(self, model, step):
        self.wandb.log({"model": model}, step=step)

    def log_artifact(self, artifact, name):
        self.wandb.log_artifact(artifact, name=name)

    def log_artifact_file(self, file, name):
        self.wandb.log_artifact(self.wandb.Artifact(name, type="file", file=file))

    def log_artifact_dir(self, dir, name):
        self.wandb.log_artifact(self.wandb.Artifact(name, type="dir", path=dir))

    def log_artifact_image(self, image, name):
        self.wandb.log_artifact(self.wandb.Image(image, caption=name))

    def log_artifact_video(self, video, name):
        self.wandb.log_artifact(self.wandb.Video(video, caption=name))

    def log_artifact_audio(self, audio, name):
        self.wandb.log_artifact(self.wandb.Audio(audio, caption=name))

    def log_artifact_html(self, html, name):
        self.wandb.log_artifact(self.wandb.Html(html, caption=name))

    def log_artifact_table(self, table, name):
        self.wandb.log_artifact(self.wandb.Table(table, caption=name))

    def log_artifact_plot(self, plot, name):
        self.wandb.log_artifact(self.wandb.Plot(plot, caption=name))

    def log_artifact_matplot(self, matplot, name):
        self.wandb.log_artifact(self.wandb.Matplot(matplot, caption=name))

