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
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import typing as t

DAY_1_FOLDER = Path("./data/day1_unsilenced")
DAY_2_FOLDER = Path("./data/day2_unsilenced")
FILE_PATTERN = re.compile(r"^(\w+)_(\w+)_\w+_\w+_\d+_(\d+).wav$")


def binary_search_leq(values, target):
    left = 0
    right = len(values) - 1
    nearest = None
    while left <= right:
        mid = left + (right - left) // 2
        if values[mid] <= target:
            nearest = mid
            left = mid + 1
        else:
            right = mid - 1
    return nearest


class RecordingDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        mic_pairs: dict[str, str],
        *,
        near_is_input: bool,
        chunk_length: int = 2048,
    ):
        self.data_path = data_path
        self.mic_pairs = mic_pairs
        self.near_is_input = near_is_input
        self.chunk_length = chunk_length

        self.input_files = []
        self.target_files = []

        files_per_input_offset = defaultdict(list)
        for file_path in data_path.iterdir():
            match = FILE_PATTERN.match(file_path.name)
            if match is None:
                continue
            (name, near_or_far, offset) = match.groups()
            files = files_per_input_offset[(name, near_or_far)]
            files.append((file_path, offset))

        # Two anti-patterns in one!
        for files in files_per_input_offset.values():
            files.sort(key=lambda x: int(x[1]))  # offset
            for i in range(len(files)):
                files[i] = files[i][0]  # name

        for near_name, far_name in mic_pairs.items():
            near_files = files_per_input_offset[(near_name, "near")]
            far_files = files_per_input_offset[(far_name, "far")]

            if near_is_input:
                input_files, target_files = near_files, far_files
            else:
                input_files, target_files = far_files, near_files

            self.input_files.extend(input_files)
            self.target_files.extend(target_files)

        self._input_markers = [0]
        self._target_markers = [0]

        # Remainder accumulators.
        self._input_loss = 0
        self._target_loss = 0

        input_marker = 0
        for input_file in self.input_files:
            with sf.SoundFile(input_file, "r") as f:
                self._input_loss += f.frames % chunk_length
                input_marker += f.frames // chunk_length
                self._input_markers.append(input_marker)

        target_marker = 0
        for target_file in self.target_files:
            with sf.SoundFile(target_file, "r") as f:
                self._target_loss += f.frames % chunk_length
                target_marker += f.frames // chunk_length
                self._target_markers.append(target_marker)

    def __getitem__(self, marker: int):
        file_index = binary_search_leq(self._input_markers, marker)
        chunk_relative = marker - self._input_markers[file_index]

        input_file = self.input_files[file_index]
        with sf.SoundFile(input_file, "r") as f:
            frame_index = chunk_relative * self.chunk_length
            f.seek(frame_index)
            input_audio = f.read(self.chunk_length)

        target_file = self.target_files[file_index]
        with sf.SoundFile(target_file, "r") as f:
            frame_index = chunk_relative * self.chunk_length
            f.seek(frame_index)
            target_audio = f.read(self.chunk_length)

        if self.near_is_input:
            near_is_input = torch.Tensor([0.0]).unsqueeze(0)
        else:
            near_is_input = torch.Tensor([1.0]).unsqueeze(0)

        return (
            torch.Tensor(input_audio).unsqueeze(0),
            torch.Tensor(target_audio).unsqueeze(0),
            near_is_input,
        )

    def __len__(self) -> int:
        i = self._input_markers[-1]
        t = self._target_markers[-1]
        assert i == t
        return i


class DistanceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        *,
        near_is_input: bool = True,
        chunk_length: int = 2048,
        shuffle: bool = True,
        batch_size: int = 64,
    ):
        super().__init__()
        self.data_path = data_path
        self.near_is_input = near_is_input
        self.chunk_length = chunk_length

        self.shuffle = shuffle
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.training_dataset = RecordingDataset(
            self.data_path,
            {"67": "269", "87": "87", "103": "103"},
            near_is_input=self.near_is_input,
            chunk_length=self.chunk_length,
        )

        self.validation_dataset = RecordingDataset(
            self.data_path,
            {"414": "414"},
            near_is_input=self.near_is_input,
            chunk_length=self.chunk_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
        )


if __name__ == "__main__":
    per_second = RecordingDataset(
        DAY_1_FOLDER, {"67": "269"}, near_is_input=False, chunk_length=44100
    )

    input_frames = 0
    for input_file in per_second.input_files:
        with sf.SoundFile(input_file, "r") as f:
            input_frames += f.frames
    print("Lossless length in seconds:", input_frames / 44100)

    print("Available length in seconds:", len(per_second))
    apparent_loss = per_second._target_loss / 44100
    print("Apparent loss in seconds", apparent_loss)

    actual_frames = 0
    for i in range(len(per_second)):
        input_audio, _, near_is_input = per_second[i]
        print(input_audio.shape)
        actual_frames += len(input_audio)
    print("Actual length in seconds:", actual_frames / 44100)
    print("Actual length with loss:", actual_frames / 44100 + apparent_loss)
