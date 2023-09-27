import torchaudio
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader


class RecordingDataset(Dataset):
    def __init__(
        self,
        *,
        stride_length: int = 1024,
        chunk_length: int = 2048,
        half: bool = True,
    ):
        self.chunk_length = chunk_length
        self.stride_length = stride_length
        self.num_frames = torchaudio.info(f"nt.wav").num_frames
        self.half = half

    def __getitem__(self, marker: int):
        with sf.SoundFile(f"nt.wav", "r") as f:
            frame_index = self.stride_length * marker
            f.seek(frame_index)
            input_audio = f.read(self.chunk_length, dtype="float32", always_2d=True)
            input_audio = torch.tensor(input_audio.T)

        with sf.SoundFile(f"67.wav", "r") as f:
            frame_index = self.stride_length * marker
            f.seek(frame_index)
            target_audio = f.read(self.chunk_length, dtype="float32", always_2d=True)
            target_audio = torch.tensor(target_audio.T)

        if self.half:
            input_audio = input_audio.half()
            target_audio = target_audio.half()

        return (
            input_audio,
            target_audio,
        )

    def __len__(self) -> int:
        return (self.num_frames - self.chunk_length) // self.stride_length


class MicroChangeDataModule(pl.LightningDataModule):
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
            RecordingDataset(
                chunk_length=self.chunk_length,
                stride_length=self.stride_length,
                half=self.half,
            )
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
