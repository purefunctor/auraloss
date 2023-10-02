from argparse import ArgumentParser
from data import DAY_1_FOLDER, DAY_2_FOLDER, DistanceAugmentDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from tcn import TCNModule
import torch

torch.set_float32_matmul_precision("high")

parser = ArgumentParser("training-script")
parser.add_argument("--nblocks", help="Number of blocks")
parser.add_argument("--dilation_growth", help="Dilation growth per block")
parser.add_argument("--kernel_size", help="Kernel size per block")
parser.add_argument("--channel_width", help="Channel width per block")
parser.add_argument("--half", action="store_true", help="Use mixed-precision training")

configuration = parser.parse_args()

nblocks = int(configuration.nblocks)
dilation_growth = int(configuration.dilation_growth)
kernel_size = int(configuration.kernel_size)
channel_width = int(configuration.channel_width)
half = configuration.half
precision = "16-mixed" if half else "32-true"

model = TCNModule(
    nblocks=nblocks,
    dilation_growth=dilation_growth,
    kernel_size=kernel_size,
    channel_width=channel_width,
    lr=0.0001,
)

datamodule = DistanceAugmentDataModule(
    DAY_1_FOLDER,
    DAY_2_FOLDER,
    chunk_size=32768,
    num_workers=16,
    half=half,
    batch_size=128,
    near_is_input=True,
)

wandb_logger = WandbLogger(
    project="near-to-far",
    name=f"TCN-{nblocks}n-{dilation_growth}g-{kernel_size}k-{channel_width}w",
    log_model="all",
)
wandb_logger.experiment.config.update(
    {
        "receptive_field": model.compute_receptive_field(),
        "batch_size": datamodule.batch_size,
        "chunk_size": datamodule.chunk_size,
    }
)

model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=1)
trainer = Trainer(
    max_epochs=20,
    callbacks=[model_checkpoint],
    precision=precision,
    logger=wandb_logger,
)

trainer.fit(model, datamodule=datamodule)
