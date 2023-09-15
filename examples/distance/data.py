"""
Implements data for distance experiment:

Input data should be stored the following folders:
* data/day1_unsilenced
* data/day2_unsilenced

File name format includes information that we need:
* 67_near_far_close_30_8.wav

In order:
* Microphone Name: 67
* Position: Far (Unused)
* Row: Close (Unused)
* Distance: 30cm
* Seconds: 8s

Note: U67 == M269

The loader's __getitem__ returns the following:
* Input Signal
* Output Signal
* Input Kind 0/Near or 1/Far, used for FiLM

The loader can be configured to either set either the
near or far signal as the input, with the other as the
output signal.
"""

from collections import defaultdict
from itertools import tee
from pathlib import Path
import pytorch_lightning as pl
import re
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

DAY_1_FOLDER = Path("./data/day1_unsilenced")
DAY_2_FOLDER = Path("./data/day2_unsilenced")
FILE_PATTERN = re.compile(r"^(\w+)_(\w+)_\w+_\w+_\d+_(\d+).wav$")


class InputTargetDataset(Dataset):
    def __init__(
        self,
        input_file: Path,
        target_file: Path,
        *,
        chunk_size: int = 2048,
        stride_factor: int = 2,
        half: bool = False,
    ):
        self.input_file = input_file
        self.target_file = target_file
        self.chunk_size = chunk_size
        self.stride_factor = stride_factor
        self.stride_length = chunk_size // stride_factor
        self.half = half

    def __getitem__(self, index):
        frame_index = index * self.stride_length

        with sf.SoundFile(self.input_file, "r") as f:
            f.seek(frame_index)
            input_audio = f.read(
                frames=self.chunk_size,
                dtype="float32",
                always_2d=True,
                fill_value=0.0,
            )
            input_audio = torch.tensor(input_audio.T)

        with sf.SoundFile(self.input_file, "r") as f:
            f.seek(frame_index)
            target_audio = f.read(
                frames=self.chunk_size,
                dtype="float32",
                always_2d=True,
                fill_value=0.0,
            )
            target_audio = torch.tensor(target_audio.T)

        if self.half:
            input_audio = input_audio.half()
            target_audio = target_audio.half()

        return (input_audio, target_audio)

    def __len__(self):
        with sf.SoundFile(self.input_file, "r") as f:
            frames = f.frames
        length = frames // self.stride_length
        if self.stride_length > 1:
            length += 1
        return length


class DistanceAugmentDataset(Dataset):
    def __init__(
        self,
        files: Path,
        pairs: dict[str, str],
        *,
        near_is_input: bool = True,
        chunk_size: int = 2048,
        stride_factor: int = 2,
        half: bool = False,
    ):
        self.files = files
        self.pairs = pairs
        self.near_is_input = near_is_input
        self.chunk_size = chunk_size
        self.stride_factor = stride_factor
        self.half = half

        files_per_microphone = defaultdict(list)
        for file in self.files.iterdir():
            match = FILE_PATTERN.match(file.name)
            if match is None:
                continue
            (microphone, near_or_far, offset) = match.groups()
            files_per_microphone[(microphone, near_or_far)].append((file, offset))

        for microphone_files in files_per_microphone.values():
            microphone_files.sort(key=lambda x: int(x[1]))  # offset
            for i in range(len(microphone_files)):
                microphone_files[i] = microphone_files[i][0]  # name

        input_target_datasets = []
        for near_microphone, far_microphone in pairs.items():
            near_files = files_per_microphone[(near_microphone, "near")]
            far_files = files_per_microphone[(far_microphone, "far")]

            if self.near_is_input:
                input_files, target_files = near_files, far_files
            else:
                target_files, input_files = near_files, far_files

            for input_file, target_file in zip(input_files, target_files):
                input_target_datasets.append(
                    InputTargetDataset(
                        input_file,
                        target_file,
                        chunk_size=self.chunk_size,
                        stride_factor=self.stride_factor,
                        half=self.half,
                    )
                )

        self.input_target_datasets = ConcatDataset(input_target_datasets)

    def __getitem__(self, index):
        return self.input_target_datasets[index]

    def __len__(self):
        return len(self.input_target_datasets)


class DistanceAugmentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        day_1_path: Path,
        day_2_path: Path,
        *,
        near_is_input: bool = True,
        chunk_size: int = 2048,
        stride_factor: int = 2,
        half: bool = False,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.day_1_path = day_1_path
        self.day_2_path = day_2_path
        self.near_is_input = near_is_input
        self.chunk_size = chunk_size
        self.stride_factor = stride_factor
        self.half = half

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        training_dataset = [
            DistanceAugmentDataset(
                files,
                {"67": "269", "87": "87", "103": "103"},
                near_is_input=self.near_is_input,
                chunk_size=self.chunk_size,
                stride_factor=self.stride_factor,
                half=self.half,
            )
            for files in [self.day_1_path, self.day_2_path]
        ]
        self.training_dataset = ConcatDataset(training_dataset)

        validation_dataset = [
            DistanceAugmentDataset(
                files,
                {"414": "414"},
                near_is_input=self.near_is_input,
                chunk_size=self.chunk_size,
                stride_factor=self.stride_factor,
                half=self.half,
            )
            for files in [self.day_1_path, self.day_2_path]
        ]
        self.validation_dataset = ConcatDataset(validation_dataset)

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
