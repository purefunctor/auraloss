import torchaudio
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from itertools import groupby

from control_data_experiment_1 import RAW_SAMPLES_DAY_1, CM1AAttack, CM1ARatio, CM1ARelease, RAW_SAMPLES_DAY_2, RAW_CM1A1_DAY_1, RAW_CM1A2_DAY_1, RAW_CM1A1_DAY_2, RAW_CM1A2_DAY_2
import subprocess
import os
CM1As = [
    [RAW_CM1A1_DAY_1, RAW_CM1A1_DAY_2, ["67_near.wav", "67_CM1A_1.wav"]],
    [RAW_CM1A2_DAY_1, RAW_CM1A2_DAY_2, ["67_near.wav", "67_CMA1_2.wav"]],
]
INDEXED = []
FILEZ = """2023-09-14 15:44:04 2243701696 103_far.wav
2023-09-14 15:44:04 2243701696 103_middle.wav
2023-09-14 15:44:10 2243701696 103_near.wav
2023-09-14 15:44:21 2243701696 269_far.wav
2023-09-14 15:44:36 2243701696 4040_middle.wav
2023-09-14 15:51:21 2243701696 414_far.wav
2023-09-14 15:51:35 2243701696 414_near.wav
2023-09-14 15:51:36 2243701696 67_1178_1.wav
2023-09-14 15:51:55 2243701696 67_1178_2.wav
2023-09-14 15:52:16 2243701696 67_CM1A_1.wav
2023-09-14 15:59:01 2243701696 67_CMA1_2.wav
2023-09-14 15:59:12 2243701696 67_MX20_1.wav
2023-09-14 15:59:15 2243701696 67_MX20_2.wav
2023-09-14 15:59:35 2243701696 67_near.wav
2023-09-14 15:59:54 2243701696 87_far.wav
2023-09-14 16:06:41 2243701696 87_near.wav
2023-09-14 16:06:51 2243701696 nt1_middle.wav"""
for WHICH_CM1A, GOODS in enumerate(CM1As):
    for FI in GOODS[2]:
        for day, (arr1, arr2) in enumerate(
            [(RAW_SAMPLES_DAY_1, GOODS[0]), (RAW_SAMPLES_DAY_2, GOODS[1])]
        ):
            subprocess.call(
                f"aws s3 cp s3://meeshkan-datasets/raw-audio/day{day+1}/{FI} {FI}",
                shell=True,
            )
            for n in range(len(arr1)):
                if arr2[n][1] == None:
                    continue
                INFI = FI
                NAME = arr2[n][0]
                RATIO = arr2[n][1][0]
                ATTACK = arr2[n][1][1]
                RELEASE = arr2[n][1][2]
                OFI = (
                    FI.split(".")[0]
                    + f"_day-{day+1}_cm1a-{WHICH_CM1A}_ratio-{RATIO}_attack-{ATTACK}_release-{RELEASE}_{NAME}_{arr1[n]}_{arr1[n+1]}.wav"
                )
            INDEXED.append((RATIO, ATTACK, RELEASE, day, WHICH_CM1A, arr1[n], arr1[n+1], INFI, OFI))

INDEXEDG = [list(x) for _, x in groupby(INDEXED, key=lambda item: (item[3], item[4], item[5], item[6]))]

def eleven78_to_0_1(eleven78):
    return (0 if eleven78 == CM1ARatio.TWO else 1 if eleven78 == CM1ARatio.THREE else 2 if eleven78 == CM1ARatio.FOUR else 3 if eleven78 == CM1ARatio.FIVE else 4 if eleven78 == CM1ARatio.SIX else 5) / 5

def attack_to_0_1(attack):
    return (0 if attack == CM1AAttack.SLOW else 1 if attack == CM1AAttack.MEDIUM_SLOW else 2 if attack == CM1AAttack.MEDIUM else 3 if attack == CM1AAttack.MEDIUM_FAST else 4) / 4

def release_to_0_1(release):
    return (0 if release == CM1ARelease.SLOW else 1 if release == CM1ARelease.MEDIUM_SLOW else 2 if release == CM1ARelease.MEDIUM else 3 if release == CM1ARelease.MEDIUM_FAST else 4) / 4

class RecordingDataset(Dataset):
    def __init__(
        self,
        eleven78: CM1ARatio,
        attack: CM1AAttack,
        release: CM1ARelease,
        filenamei: str,
        filenameo: str,
        *,
        chunk_length: int = 2048,
        stride_length: int = 1024,
        half: bool = True,
    ):
        self.eleven78 = eleven78
        self.attack = attack
        self.release = release
        self.filenamei = filenamei
        self.filenameo = filenameo
        self.chunk_length = chunk_length
        self.stride_length = stride_length
        self.num_frames = torchaudio.info(filenameo).num_frames
        self.half = half

    def __getitem__(self, marker: int):
            frame_index = self.stride_length * marker
            with sf.SoundFile(self.filenamei, "r") as f:
                f.seek(frame_index)
                input_audio = f.read(frames=self.chunk_length, dtype="float32", always_2d=True, fill_value=0.0)
                input_audio = torch.tensor(input_audio.T)
            with sf.SoundFile(self.filenameo, "r") as f:
                f.seek(frame_index)
                target_audio = f.read(frames=self.chunk_length, dtype="float32", always_2d=True, fill_value=0.0)
                target_audio = torch.tensor(target_audio.T)
            parameters = torch.tensor([[eleven78_to_0_1(self.eleven78), attack_to_0_1(self.attack), release_to_0_1(self.release)]])
            
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
            RecordingDataset(I[0][0],
        I[0][1],
        I[0][2],
        I[0][7] if '67_near' in I[0][7] else I[1][7],
        I[1][7] if '67_near' in I[0][7] else I[0][7], chunk_length=self.chunk_length, stride_length = self.stride_length, half=self.half)
            for I in INDEXEDG 
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

if __name__ == '__main__':
    import subprocess
    for FI in INDEXED:
        subprocess.call(f'aws s3 cp s3://meeshkan-datasets/compressor-cm1a/{FI[-1]} {FI[-1]}', shell=True)
