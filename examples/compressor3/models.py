import auraloss
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from efficient_net import EfficientNet
from mobilenetv2 import MobileNetV2
from tcn import TCNModel, center_crop
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, *, embed_dim: int, width_mult: int, encoder_model: str):
        super().__init__()

        self.embed_dim = embed_dim
        self.width_mult = width_mult
        self.encoder_model = encoder_model

        if encoder_model == "mobilenet_v2":
            self.encoder = MobileNetV2(embed_dim=embed_dim, width_mult=width_mult)
        elif encoder_model == "efficient_net":
            self.encoder = EfficientNet.from_name(
                "efficientnet-b2",
                in_channels=1,
                image_size=(128, 65),
                include_top=False,
            )
            self.embedding_projection = nn.Conv2d(
                in_channels=1408,
                out_channels=embed_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            )
        else:
            raise ValueError(f"Invalid encoder model {encoder_model}")

        self.window = nn.Parameter(torch.hann_window(4096))

    def forward(self, x):
        batch_size, channels, samples = x.size()

        X = torch.stft(
            x.view(batch_size, -1), 4096, 2048, window=self.window, return_complex=True
        )

        X_db = torch.pow(X.abs() + 1e-8, 0.3)
        X_db_norm = X_db

        X_min = X_db.min()
        X_max = X_db.max()

        X_db_norm = (X_db_norm - X_min) / X_max
        X_db_norm = X_db_norm.unsqueeze(1).permute(0, 1, 3, 2)

        if self.encoder_model == "mobilenet_v2":
            X_db_norm = X_db_norm.repeat(1, 3, 1, 1)

            e = self.encoder(X_db_norm)
            e = F.adaptive_avg_pool2d(e, 1).reshape(e.shape[0], -1)

            norm = torch.norm(e, p=2, dim=-1, keepdim=True)
            e_norm = e / norm

        else:
            e = self.encoder(X_db_norm)

            e = self.embedding_projection(e)
            e = torch.squeeze(e, dim=3)
            e = torch.squeeze(e, dim=2)

            norm = torch.norm(e, p=2, dim=-1, keepdim=True)
            e_norm = e / norm

        return e_norm


class Controller(nn.Module):
    def __init__(self, *, parameter_count, encoder_dimensions, hidden_dimensions):
        super().__init__()

        self.parameter_count = parameter_count
        self.encoder_dimensions = encoder_dimensions
        self.hidden_dimensions = hidden_dimensions

        self.mlp = nn.Sequential(
            nn.Linear(encoder_dimensions * 2, hidden_dimensions),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dimensions, parameter_count),
            nn.Sigmoid(),
        )

    def forward(self, e_x, e_y):
        e_xy = torch.cat((e_x, e_y), dim=-1)
        p = self.mlp(e_xy).unsqueeze(1)

        return p


class System(pl.LightningModule):
    def __init__(
        self,
        *,
        embedding_dimensions=1024,
        width_multiplier=1,
        parameter_count=3,
        encoder_model="efficient_net",
        lr=0.001,
    ):
        super().__init__()

        self.embedding_dimensions = embedding_dimensions
        self.width_multiplier = width_multiplier
        self.parameter_count = parameter_count
        self.encoder_model = encoder_model
        self.lr = lr

        self.encoder = Encoder(
            embed_dim=embedding_dimensions,
            width_mult=width_multiplier,
            encoder_model=encoder_model,
        )
        self.controller = Controller(
            parameter_count=parameter_count,
            encoder_dimensions=embedding_dimensions,
            hidden_dimensions=256,
        )
        self.processor = TCNModel.load_from_checkpoint(
            "cma1-compressor-unsilenced/1i2bon8e/checkpoints/epoch=19-step=48900.ckpt"
        )

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.esr = auraloss.time.ESRLoss()
        self.dc = auraloss.time.DCLoss()
        self.logcosh = auraloss.time.LogCoshLoss()
        self.sisdr = auraloss.time.SISDRLoss()
        self.stft = auraloss.freq.STFTLoss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, x, y):
        e_x = self.encoder(x)
        e_y = self.encoder(y)
        p_h = self.controller(e_x, e_y)
        y_h = self.processor(x, p_h)
        return y_h, p_h

    def training_step(self, batch, batch_idx):
        input_signal, target_signal, _ = batch

        predicted_signal, _ = self(input_signal, target_signal)

        target_signal = center_crop(target_signal, predicted_signal.shape)
        predicted_signal = center_crop(predicted_signal, predicted_signal.shape)

        s_loss = self.mrstft(predicted_signal, target_signal)

        self.log(
            "train_loss",
            s_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return s_loss

    def validation_step(self, batch, batch_idx):
        input_signal, target_signal, _ = batch

        predicted_signal, _ = self(input_signal, target_signal)

        target_signal = center_crop(target_signal, predicted_signal.shape)
        predicted_signal = center_crop(predicted_signal, predicted_signal.shape)

        l1_loss = self.l1(predicted_signal, target_signal)
        esr_loss = self.esr(predicted_signal, target_signal)
        dc_loss = self.dc(predicted_signal, target_signal)
        logcosh_loss = self.logcosh(predicted_signal, target_signal)
        sisdr_loss = self.sisdr(predicted_signal, target_signal)
        stft_loss = self.stft(predicted_signal, target_signal)
        mrstft_loss = self.mrstft(predicted_signal, target_signal)

        aggregate_loss = (
            l1_loss
            + esr_loss
            + dc_loss
            + logcosh_loss
            + sisdr_loss
            + mrstft_loss
            + stft_loss
        )

        self.log("val_loss", aggregate_loss, sync_dist=True)
        self.log("val_loss/L1", l1_loss, sync_dist=True)
        self.log("val_loss/ESR", esr_loss, sync_dist=True)
        self.log("val_loss/DC", dc_loss, sync_dist=True)
        self.log("val_loss/LogCosh", logcosh_loss, sync_dist=True)
        self.log("val_loss/SI-SDR", sisdr_loss, sync_dist=True)
        self.log("val_loss/STFT", stft_loss, sync_dist=True)
        self.log("val_loss/MRSTFT", mrstft_loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=4, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    from data import CompressorDataModule

    torch.set_float32_matmul_precision("high")

    datamodule = CompressorDataModule(
        chunk_length=32768 * 2,
        stride_length=32768 * 2,
        batch_size=16,
        num_workers=16,
        shuffle=True,
        half=True,
    )

    system = System(
        embedding_dimensions=4096,
        width_multiplier=1,
        parameter_count=3,
        encoder_model="efficient_net",
    )

    wandb_logger = WandbLogger(project="deepafx-st", log_model="all")
    model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_epochs=1)
    trainer = pl.Trainer(
        max_epochs=10,
        precision="16-mixed",
        callbacks=[model_checkpoint],
        logger=wandb_logger,
    )
    trainer.fit(system, datamodule=datamodule)
