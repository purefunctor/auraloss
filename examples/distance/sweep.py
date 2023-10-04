from data import DAY_1_FOLDER, DAY_2_FOLDER, DistanceAugmentDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from tcn import TCNModule
import torch
from torchinfo import summary
import wandb

torch.set_float32_matmul_precision("high")


def train_model():
    wandb.init()
    config = wandb.config

    nblocks = config.nblocks
    dilation_growth = config.dilation_growth
    kernel_size = config.kernel_size
    channel_width = config.channel_width
    stack_size = config.stack_size
    batch_size = config.batch_size

    model = TCNModule(
        nblocks=nblocks,
        dilation_growth=dilation_growth,
        kernel_size=kernel_size,
        channel_width=channel_width,
        stack_size=stack_size,
        lr=0.002,
    )
    info = summary(model, verbose=0)
    parameter_count = info.total_params

    if model.compute_receptive_field() > 32768:
        print(f"Too large: {nblocks}n-{dilation_growth}g-{kernel_size}k-{channel_width}w-{stack_size}s-{parameter_count}p")
        return

    datamodule = DistanceAugmentDataModule(
        DAY_1_FOLDER,
        DAY_2_FOLDER,
        chunk_size=32768,
        num_workers=16,
        half=True,
        batch_size=batch_size,
        near_is_input=True,
    )

    wandb_logger = WandbLogger(
        project="near-to-far",
        name=f"TCN-{nblocks}n-{dilation_growth}g-{kernel_size}k-{channel_width}w-{stack_size}s-{parameter_count}p",
        log_model="all",
    )

    model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=1)
    trainer = Trainer(
        max_epochs=5,
        callbacks=[model_checkpoint],
        precision="16-mixed",
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "name": "near-to-far-tiny-sweep",
        "metric": {
            "goal": "minimize",
            "name": "val_loss/MRSTFT",
        },
        "parameters": {
            "nblocks": {"min": 4, "max": 10},
            "dilation_growth": {"min": 2, "max": 10},
            "kernel_size": {"min": 5, "max": 25},
            "channel_width": {"min": 32, "max": 192},
            "stack_size": {"min": 2, "max": 10},
            "batch_size": {"values": [16, 32, 64]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="near-to-far")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=10)
