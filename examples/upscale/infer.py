import soundfile as sf
from tcn import TCNModule
import torch
import wandb


api = wandb.Api()

artifact = api.artifact("meeshkan/enhancement/model-i9lviph4:v19")
weights = artifact.get_path("model.ckpt").download("/tmp")
model = TCNModule.load_from_checkpoint(weights, map_location=torch.device("cpu")).eval()
receptive_field = model.compute_receptive_field()

with sf.SoundFile("data/day1_unsilenced/67_near_far_close_30_1192.wav", "r") as f:
    f.seek(88 * 44100 - receptive_field)
    input_audio = f.read(44100 * 10 + receptive_field, dtype="float32", always_2d=True)
    input_audio = torch.tensor(input_audio.T).unsqueeze(0)

with sf.SoundFile("data/day1_unsilenced/nt1_middle_far_mid_48_1192.wav", "r") as f:
    f.seek(88 * 44100 - receptive_field)
    target_audio = f.read(44100 * 10 + receptive_field, dtype="float32", always_2d=True)
    target_audio = torch.tensor(target_audio.T).unsqueeze(0)

y = model(input_audio).squeeze().detach().numpy()

sf.write("nt1.wav", input_audio.squeeze().numpy()[receptive_field:], samplerate=44100)
sf.write("u67.wav", target_audio.squeeze().numpy()[receptive_field:], samplerate=44100)
sf.write("ups.wav", y, samplerate=44100)
