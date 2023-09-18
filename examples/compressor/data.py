from control import MX20
import torchaudio
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

torchaudio.set_audio_backend("soundfile")

class RecordingDataset(Dataset):
    def __init__(
        self,
        mx20: MX20,
        *,
        chunk_length: int = 2048,
        stride_length: int = 1024,
        half: bool = True,
    ):
        self.mx20 = mx20
        self.chunk_length = chunk_length
        self.stride_length = stride_length
        self.num_frames = torchaudio.info(f"o_x_{mx20}.wav").num_frames
        self.half = half

    def __getitem__(self, marker: int):
            frame_index = self.stride_length * marker
            input_audio, _ = torchaudio.load(f"o_x_{self.mx20}.wav", frame_offset=frame_index,num_frames=self.chunk_length)
            target_audio, _ = torchaudio.load(f"o_y_{self.mx20}.wav", frame_offset=frame_index,num_frames=self.chunk_length)
            match self.mx20:
                case MX20.TWO:
                    parameters = torch.tensor([[0.2]])
                case MX20.FOUR:
                    parameters = torch.tensor([[0.4]])
                case MX20.EIGHT:
                    parameters = torch.tensor([[0.6]])
                case MX20.TWELVE:
                    parameters = torch.tensor([[0.8]])
                case _:
                    raise Exception(f"Invalid parameters {self.mx20}")

            if self.half:
                input_audio = input_audio.half()
                target_audio = target_audio.half()
                parameters = parameters.half()

            return (
                input_audio,
                target_audio,
                parameters,
            )
    def __len__(self) -> int:
        return (self.num_frames - self.chunk_length) // self.stride_length


class CompressorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        chunk_length: int = 2048,
        stride_length: int = 1024,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = True,
        half: bool = True,
    ):
        super().__init__()
        self.chunk_length = chunk_length
        self.stride_length = stride_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.half = half

    def setup(self, stage: str):
        datasets = [
            RecordingDataset(mx20, chunk_length=self.chunk_length, stride_length = self.stride_length, half=self.half)
            for mx20 in (MX20.TWO, MX20.FOUR, MX20.EIGHT, MX20.TWELVE)
        ]
        dataset = ConcatDataset(datasets)
        training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [0.8, 0.2]
        )
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
