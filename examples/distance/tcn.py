import auraloss
from pathlib import Path
import pytorch_lightning as pl
import soundfile as sf
import torch


def center_crop(x, shape):
    start = (x.shape[-1] - shape[-1]) // 2
    stop = start + shape[-1]
    return x[..., start:stop]


class FiLM(torch.nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x


class TCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        padding=0,
        dilation=1,
        depthwise=False,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.depthwise = depthwise

        groups = out_ch if depthwise and (in_ch % out_ch == 0) else 1

        self.conv1 = torch.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        if depthwise:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        self.film = FiLM(out_ch, 128)

        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(
            in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False
        )

    def forward(self, x, p=None):
        x_in = x

        x = self.conv1(x)
        if self.depthwise:
            x = self.conv1b(x)
        x = self.film(x, p)
        x = self.relu(x)

        x_res = self.res(x_in)
        x = x + center_crop(x_res, x.shape)

        return x


class TCNModule(pl.LightningModule):
    """Temporeal convolutional network with conditioning module."""

    def __init__(
        self,
        nparams: int = 1,
        nblocks: int = 10,
        kernel_size: int = 3,
        dilation_growth: int = 1,
        channel_width: int = 64,
        stack_size: int = 10,
        depthwise: bool = False,
        lr=0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()

        if nparams == 0:
            raise ValueError("Must have at least one conditioning parameter.")

        self.loss_function = auraloss.freq.MultiResolutionSTFTLoss()

        self.expand_parameters = torch.nn.Sequential(
            torch.nn.Linear(nparams, 32),
            torch.nn.PReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.PReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.PReLU(),
        )

        self.tcn_blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = channel_width if n > 0 else 1
            out_ch = channel_width
            dilation = dilation_growth ** (n % stack_size)
            self.tcn_blocks.append(
                TCNBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    depthwise=depthwise,
                )
            )

        self.reduce_output = torch.nn.Conv1d(out_ch, 1, kernel_size=1)
        self.final_activation = torch.nn.Tanh()

        self.validation_epoch_outputs = []

    def forward(self, x, p):
        conditioning = self.expand_parameters(p)
        for index, tcn_block in enumerate(self.tcn_blocks):
            x = tcn_block(x, conditioning)
            if index == 0:
                skips = x
            else:
                skips = center_crop(skips, x.shape) + x
        x = self.reduce_output(x + skips)
        x = self.final_activation(x)
        return x

    def training_step(self, batch, *_):
        _, input_signal, target_signal, parameters = batch

        predicted_signal = self(input_signal, parameters)
        target_signal = center_crop(target_signal, predicted_signal.shape)

        loss = self.loss_function(predicted_signal, target_signal)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, *_):
        file_name, input_signal, target_signal, parameters = batch
        predicted_signal = self(input_signal, parameters)

        input_signal = center_crop(input_signal, predicted_signal.shape)
        target_signal = center_crop(target_signal, predicted_signal.shape)

        loss = self.loss_function(predicted_signal, target_signal)

        self.log("val_loss", loss, sync_dist=True)

        self.validation_epoch_outputs.append(
            {
                "name": file_name,
                "input": input_signal.cpu().numpy(),
                "target": target_signal.cpu().numpy(),
                "prediction": predicted_signal.cpu().numpy(),
                "parameters": parameters.cpu().numpy(),
            }
        )

    def on_validation_epoch_end(self):
        outputs = {
            "name": [],
            "input": [],
            "target": [],
            "prediction": [],
            "parameters": [],
        }
        for validation_epoch_output in self.validation_epoch_outputs:
            outputs["name"].append(validation_epoch_output["name"])
            outputs["input"].append(validation_epoch_output["input"])
            outputs["target"].append(validation_epoch_output["target"])
            outputs["prediction"].append(
                validation_epoch_output["prediction"]
            )
            outputs["parameters"].append(
                validation_epoch_output["parameters"]
            )
        audio_output_folder = (
            Path("lightning_logs") / f"version_{self.logger.version}" / "output"
        )
        if not audio_output_folder.exists():
            audio_output_folder.mkdir()
        for names, inputs in zip(outputs["name"], outputs["input"]):
            for n, i in zip(names, inputs):
                print(i.shape)
                audio_file = audio_output_folder / n
                if not audio_file.exists():
                    sf.write(audio_file, [], samplerate=44100)
                with sf.SoundFile(audio_file, "r+") as f:
                    f.seek(0, sf.SEEK_END)
                    f.write(i[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
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
    from data import DAY_1_FOLDER, DAY_2_FOLDER, DistanceDataModule
    from pytorch_lightning import Trainer
    import torch

    model = TCNModule()
    datamodule = DistanceDataModule(
        DAY_1_FOLDER, DAY_2_FOLDER, chunk_length=44100, num_workers=24
    )

    trainer = Trainer(max_epochs=10, devices=1, num_nodes=1)
    trainer.validate(model, datamodule=datamodule)
