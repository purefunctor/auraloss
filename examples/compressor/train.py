from data import CompressorDataModule
from tcn import TCNModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

half = True

if half:
    precision = "16-mixed"
else:
    precision = "32-true"

model = TCNModel(
    nparams=2,
    kernel_size=15,
    channel_width=32,
    dilation_growth=2,
    lr=0.001,
    train_loss="mrstft",
)
CHUNK_LENGTH = 32768
datamodule = CompressorDataModule(chunk_length=CHUNK_LENGTH, stride_length=CHUNK_LENGTH//4, batch_size=128, num_workers=8)

wandb_logger = WandbLogger(project="unified-compressor", log_model="all")
model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=1)
trainer = Trainer(max_epochs=20, precision=precision, callbacks=[model_checkpoint], logger=wandb_logger)

trainer.fit(model, datamodule=datamodule)
