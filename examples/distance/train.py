from argparse import ArgumentParser
from data import DAY_1_FOLDER, DAY_2_FOLDER, DistanceAugmentDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from tcn import TCNModule
import torch

torch.set_float32_matmul_precision("high")

parser = ArgumentParser("training-script")
parser.add_argument("--nblocks", help="Number of blocks")
parser.add_argument("--dilation-growth", help="Dilation growth per block")
parser.add_argument("--kernel-size", help="Kernel size per block")
parser.add_argument("--channel-width", help="Channel width per block")
parser.add_argument("--stack-size", help="Number of blocks before dilation resets")
parser.add_argument("--half", action="store_true", help="Use mixed-precision training")
parser.add_argument("--far", action="store_true", help="Train with far-to-near")

configuration = parser.parse_args()

nblocks = int(configuration.nblocks)
dilation_growth = int(configuration.dilation_growth)
kernel_size = int(configuration.kernel_size)
channel_width = int(configuration.channel_width)
stack_size = int(configuration.stack_size)
half = configuration.half
far = configuration.far
precision = "16-mixed" if half else "32-true"
project = "near-to-far" if not far else "far-to-near"

model = TCNModule(
    nblocks=nblocks,
    dilation_growth=dilation_growth,
    kernel_size=kernel_size,
    channel_width=channel_width,
    stack_size=stack_size,
    lr=0.002,
)

datamodule = DistanceAugmentDataModule(
    DAY_1_FOLDER,
    DAY_2_FOLDER,
    chunk_size=32768,
    num_workers=16,
    half=half,
    batch_size=64,
    near_is_input=not far,
)

wandb_logger = WandbLogger(
    project=project,
    name=f"TCN-{nblocks}n-{dilation_growth}g-{kernel_size}k-{channel_width}w-{stack_size}s",
    log_model="all",
)

if rank_zero_only.rank == 0:
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
