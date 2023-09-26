from data import DAY_1_FOLDER, DAY_2_FOLDER, EnhancementDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from tcn import TCNModule
import torch
import wandb

torch.set_float32_matmul_precision("high")

half = True
if half:
    precision = "16-mixed"
else:
    precision = "32-true"

configuration = {
    # "uTCN-100-C": {
    #     "nblocks": 4,
    #     "dilation_growth": 10,
    #     "kernel_size": 5,
    #     "causal": True,
    #     "channel_width": 32,
    #     "lr": 0.001,
    # },
    # "uTCN-300-C": {
    #     "nblocks": 4,
    #     "dilation_growth": 10,
    #     "kernel_size": 13,
    #     "channel_width": 32,
    #     "causal": True,
    #     "lr": 0.001,
    # },
    "uTCN-100-Cx3": {
        "nblocks": 4,
        "dilation_growth": 10,
        "kernel_size": 5,
        "channel_width": 96,
        "causal": True,
        "lr": 0.001,
    },
    "uTCN-300-Cx3": {
        "nblocks": 4,
        "dilation_growth": 10,
        "kernel_size": 13,
        "channel_width": 96,
        "causal": True,
        "lr": 0.001,
    },
    # "TCN-324-C": {
    #     "nblocks": 10,
    #     "dilation_growth": 2,
    #     "kernel_size": 15,
    #     "channel_width": 32,
    #     "causal": True,
    #     "lr": 0.001,
    # },
    "TCN-324-Cx3": {
        "nblocks": 10,
        "dilation_growth": 2,
        "kernel_size": 15,
        "channel_width": 96,
        "causal": True,
        "lr": 0.001,
    },
}

for n, p in configuration.items():
    model = TCNModule(**p)
    datamodule = EnhancementDataModule(
        DAY_1_FOLDER,
        DAY_2_FOLDER,
        chunk_size=32768,
        num_workers=16,
        half=half,
        batch_size=128,
    )

    wandb_logger = WandbLogger(project="enhancement", name=f"{n}", log_model="all")
    wandb_logger.experiment.config.update({
        "receptive_field": model.compute_receptive_field(),
        "batch_size": datamodule.batch_size,
        "chunk_size": datamodule.chunk_size,
    })

    model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=1)
    trainer = Trainer(
        max_epochs=20,
        callbacks=[model_checkpoint],
        precision=precision,
        logger=wandb_logger,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )

    wandb.finish()
