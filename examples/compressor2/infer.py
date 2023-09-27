import soundfile as sf
from data import Eleven78, eleven78_to_0_1, attack_to_0_1, release_to_0_1
from tcn import TCNModel
import torch
import wandb

api = wandb.Api()

artifact = api.artifact("meeshkan/eleven78-compressor-unsilenced/model-yeasc3a8:v39")
weights = artifact.get_path("model.ckpt").download("/tmp/")
model = TCNModel.load_from_checkpoint(weights).eval()

peak_index = 32165376

input_file = "67_near_day1_ratio4_attack7_release5_Aku_542232920_579457368.wav"
target_file = "67_1178_2_day1_ratio4_attack7_release5_Aku_542232920_579457368.wav"

with sf.SoundFile(input_file, "r") as f:
    f.seek(peak_index - 44100 * 5)
    input_audio = f.read(44100 * 10, dtype="float32", always_2d=True)
    sf.write("input.wav", input_audio, 44100)

with sf.SoundFile(target_file, "r") as f:
    f.seek(peak_index - 44100 * 5)
    target_audio = f.read(44100 * 10, dtype="float32", always_2d=True)
    sf.write("target.wav", target_audio, 44100)

input_audio = torch.tensor(input_audio.T).reshape(1, 1, -1).cuda()
target_audio = torch.tensor(target_audio.T).reshape(1, 1, -1).cuda()

for c in [Eleven78.FOUR, Eleven78.EIGHT, Eleven78.TWELVE]:
    parameters = torch.tensor([[[
        eleven78_to_0_1(c),
        attack_to_0_1(7),
        release_to_0_1(5),
    ]]]).cuda()

    predicted_audio = model(input_audio, parameters)

    sf.write(f"predicted_{c}.wav", predicted_audio.squeeze().detach().cpu().numpy(), 44100)
