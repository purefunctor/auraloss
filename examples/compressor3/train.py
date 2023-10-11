from data import CompressorDataModule
from tcn import TCNModel
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

torch.set_float32_matmul_precision("high")

half = True

if half:
    precision = "16-mixed"
else:
    precision = "32-true"

model = TCNModel(
    nparams=3,
    nblocks=10,
    kernel_size=15,
    channel_width=32,
    dilation_growth=2,
    lr=0.001,
    train_loss="mrstft",
)
CHUNK_LENGTH = 32768
datamodule = CompressorDataModule(
    chunk_length=CHUNK_LENGTH,
    stride_length=CHUNK_LENGTH // 2,
    batch_size=64,
    num_workers=16,
)

wandb_logger = WandbLogger(project="cma1-compressor-unsilenced", log_model="all")
model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=1)
trainer = Trainer(
    max_epochs=20,
    precision=precision,
    callbacks=[model_checkpoint],
    logger=wandb_logger,
)

trainer.fit(model, datamodule=datamodule)