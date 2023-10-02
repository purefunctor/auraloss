from data import (
    CM1ARatio,
    CM1AAttack,
    CM1ARelease,
    ratio_to_0_1,
    attack_to_0_1,
    release_to_0_1,
)
from itertools import product
from pathlib import Path
import soundfile as sf
from tcn import TCNModel
import torch
import wandb

api = wandb.Api()

artifact = api.artifact("meeshkan/cma1-compressor-unsilenced/model-6kqf71ne:v39")
weights = artifact.get_path("model.ckpt").download("/tmp/")
model = TCNModel.load_from_checkpoint(weights).eval()

input_pairs = [
    ("allatonce.wav", "allatonce"),
    ("filtered.wav", "filtered"),
]

_results = Path("results")
if not _results.exists():
    _results.mkdir()

for input_file, base_name in input_pairs:
    with sf.SoundFile(input_file, "r") as f:
        input_audio = f.read(44100 * 30, dtype="float32", always_2d=True)
        input_audio = torch.tensor(input_audio.T).reshape(1, 1, -1).cuda()

    for ratio, attack, release in product(CM1ARatio, CM1AAttack, CM1ARelease):
        parameters = torch.tensor(
            [[[ratio_to_0_1(ratio), attack_to_0_1(attack), release_to_0_1(release)]]]
        ).cuda()

        predicted_audio = (
            model(input_audio, parameters).squeeze().detach().cpu().numpy()
        )
        sf.write(
            _results / f"{base_name}_{ratio}_{attack}_{release}.wav",
            predicted_audio,
            samplerate=44100,
        )

        del parameters

    del input_audio

    torch.cuda.empty_cache()
