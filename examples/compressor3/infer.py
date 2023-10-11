from collections import defaultdict
from data import FILE_PATTERN
from models import System
from pathlib import Path
import soundfile as sf
import torch
import wandb

api = wandb.Api()

artifact = api.artifact("meeshkan/deepafx-st/model-pheign5j:v9")
weights = artifact.get_path("model.ckpt").download("/tmp")
model = System(
    embedding_dimensions=4096,
    width_multiplier=1,
    parameter_count=3,
    encoder_model="efficient_net",
)
model.load_state_dict(torch.load(weights)["state_dict"])
model.cuda()

files_per_name_offset = {
    "day1": defaultdict(list),
    "day2": defaultdict(list),
}

_data = Path("data")
for day in ["day1", "day2"]:
    path = _data / day
    for file in path.iterdir():
        match = FILE_PATTERN.match(file.name)
        if match is None:
            continue
        (name, _, offset) = match.groups()
        files_per_name_offset[day][(name, offset)] = file

offset_start_pairs = {
    "day1": [
        ("1192", 88),
        ("7885", 30),
        ("13140", 15),
        ("4273", 15),
    ],
    "day2": [
        ("2146", 19),
        ("19066", 10),
        ("19919", 21),
        ("22727", 6),
    ],
}

input_microphones = [
    "nt1",
    # "414",
    # "4040",
]

_results = Path("results")
if not _results.exists():
    _results.mkdir()

for day in ["day1", "day2"]:
    for name in input_microphones:
        for offset, start in offset_start_pairs[day]:
            input_file = files_per_name_offset[day][(name, offset)]
            target_file = files_per_name_offset[day][("67", offset)]

            with sf.SoundFile(input_file, "r") as f:
                f.seek(start * 44100)
                input_audio = f.read(44100 * 15, dtype="float32", always_2d=True)
                sf.write(
                    f"results/{day}_{name}_{offset}_input.wav",
                    input_audio,
                    samplerate=44100,
                )
                input_audio = torch.tensor(input_audio.T).unsqueeze(0).cuda()

            with sf.SoundFile(target_file, "r") as f:
                f.seek(start * 44100)
                target_audio = f.read(44100 * 15, dtype="float32", always_2d=True)
                sf.write(
                    f"results/{day}_{name}_{offset}_target.wav",
                    target_audio,
                    samplerate=44100,
                )
                target_audio = torch.tensor(target_audio.T).unsqueeze(0).cuda()

            prediction_audio, _ = model(input_audio, target_audio)
            prediction_audio = prediction_audio.squeeze().detach().cpu().numpy()

            sf.write(
                f"results/{day}_{name}_{offset}_prediction.wav",
                prediction_audio,
                samplerate=44100,
            )

            del input_audio
            del target_audio
            del prediction_audio

            torch.cuda.empty_cache()
