import soundfile as sf
from tcn import TCNModule
import torch

tcn = TCNModule.load_from_checkpoint(
    "logs/epoch=49-step=25950.ckpt"
)
tcn.eval()

tcn.to("cuda")

with sf.SoundFile("data/day1_unsilenced/414_near_far_close_30_1192.wav", "r") as f:
    f.seek(88 * 44100)
    input_audio = f.read(44100 * 10)

with sf.SoundFile("data/day1_unsilenced/414_far_far_far_65_1192.wav", "r") as f:
    f.seek(88 * 44100)
    target_audio = f.read(44100 * 10)

i = torch.Tensor(input_audio).unsqueeze(0).unsqueeze(0).to("cuda")
y = tcn(i).squeeze().squeeze().detach().cpu().numpy()

sf.write("414_near.wav", input_audio, samplerate=44100)
sf.write("414_far.wav", target_audio, samplerate=44100)
sf.write("414_near_to_far.wav", y, samplerate=44100)
