import soundfile as sf
from tcn import TCNModule
import torch
import wandb


api = wandb.Api()

artifact = api.artifact("meeshkan/enhancement/model-tn4ghvkz:v19")
weights = artifact.get_path("model.ckpt").download("/tmp")
model = TCNModule.load_from_checkpoint(weights).eval()
receptive_field = model.compute_receptive_field()


files = [
    ("data/day1_unsilenced/nt1_middle_far_mid_48_1192.wav", "data/day1_unsilenced/67_near_far_close_30_1192.wav", 88),
    ("data/day1_unsilenced/nt1_middle_far_mid_48_7885.wav", "data/day1_unsilenced/67_near_far_close_30_7885.wav", 30),
    ("data/day1_unsilenced/nt1_middle_far_mid_48_13140.wav", "data/day1_unsilenced/67_near_far_close_30_13140.wav", 15),
    ("data/day1_unsilenced/nt1_middle_close_mid_38_4273.wav", "data/day1_unsilenced/67_near_close_close_20_4273.wav", 15),
]


for index, (input_file, target_file, offset) in enumerate(files):
    with sf.SoundFile(input_file, "r") as f:
        f.seek(offset * 44100 - receptive_field)
        input_audio = f.read(44100 * 15 + receptive_field, dtype="float32", always_2d=True)
        sf.write(f"sweep/{index}_input.wav", input_audio[receptive_field:], samplerate=44100)
        input_audio = torch.tensor(input_audio.T).unsqueeze(0).cuda()

    with sf.SoundFile(target_file, "r") as f:
        f.seek(offset * 44100 - receptive_field)
        target_audio = f.read(44100 * 15 + receptive_field, dtype="float32", always_2d=True)
        sf.write(f"sweep/{index}_target.wav", target_audio[receptive_field:], samplerate=44100)
        target_audio = torch.tensor(target_audio.T).unsqueeze(0).cuda()

    y = model(input_audio).squeeze().detach().cpu().numpy()
    sf.write(f"sweep/{index}_predicted.wav", y, samplerate=44100)
