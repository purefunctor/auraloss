import soundfile as sf
from tcn import TCNModule
import torch

tcn = TCNModule.load_from_checkpoint(
    "lightning_logs/version_2/checkpoints/epoch=49-step=32400.ckpt"
)
tcn.eval()

tcn.to("cuda")

with sf.SoundFile("data/day1_unsilenced/nt1_middle_far_mid_48_1192.wav", "r") as f:
    f.seek(88 * 44100)
    input_audio = f.read(44100 * 10)

with sf.SoundFile("data/day1_unsilenced/67_near_far_close_30_1192.wav", "r") as f:
    f.seek(88 * 44100)
    target_audio = f.read(44100 * 10)

i = torch.Tensor(input_audio).unsqueeze(0).unsqueeze(0).to("cuda")
y = tcn(i).squeeze().squeeze().detach().cpu().numpy()

sf.write("nt1_middle.wav", input_audio, samplerate=44100)
sf.write("67_near.wav", target_audio, samplerate=44100)
sf.write("nt1_to_67.wav", y, samplerate=44100)
