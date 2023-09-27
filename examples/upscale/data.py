from collections import defaultdict
from pathlib import Path
import pytorch_lightning as pl
import re
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split

DAY_1_FOLDER = Path("./data/day1_unsilenced")
DAY_2_FOLDER = Path("./data/day2_unsilenced")
FILE_PATTERN = re.compile(r"^(.+)_(\w+)_\w+_\w+_\d+_(\d+).wav$")


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


class EnhancementDataset(Dataset):
    def __init__(
        self,
        files: Path,
        pairs: dict[str, str],
        *,
        chunk_size: int = 2048,
        stride_factor: int = 2,
        half: bool = False,
    ):
        self.files = files
        self.pairs = pairs
        self.chunk_size = chunk_size
        self.stride_factor = stride_factor
        self.half = half

        files_per_microphone = defaultdict(list)
        for file in self.files.iterdir():
            match = FILE_PATTERN.match(file.name)
            if match is None:
                continue
            (microphone, _, offset) = match.groups()
            files_per_microphone[microphone].append((file, offset))

        for microphone_files in files_per_microphone.values():
            microphone_files.sort(key=lambda x: int(x[1]))  # offset
            for i in range(len(microphone_files)):
                microphone_files[i] = microphone_files[i][0]  # name

        input_target_datasets = []
        for input_microphone, target_microphone in pairs.items():
            input_files = files_per_microphone[input_microphone]
            target_files = files_per_microphone[target_microphone]

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


class EnhancementDataModule(pl.LightningDataModule):
    def __init__(
        self,
        day_1_path: Path,
        day_2_path: Path,
        *,
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
        self.chunk_size = chunk_size
        self.stride_factor = stride_factor
        self.half = half

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        dataset = ConcatDataset(
            [
                EnhancementDataset(
                    files,
                    {"nt1": "67"},
                    chunk_size=self.chunk_size,
                    stride_factor=self.stride_factor,
                    half=self.half,
                )
                for files in [self.day_1_path, self.day_2_path]
            ]
        )
        training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [0.8, 0.2]
        )

        training_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])

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
