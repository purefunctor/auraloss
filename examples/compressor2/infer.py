import soundfile as sf
from data import Eleven78, eleven78_to_0_1, attack_to_0_1, release_to_0_1
import librosa
from tcn import TCNModel
import torch
import wandb

api = wandb.Api()

artifact = api.artifact("meeshkan/eleven78-compressor-unsilenced/model-yeasc3a8:v39")
weights = artifact.get_path("model.ckpt").download("/tmp/")
model = TCNModel.load_from_checkpoint(weights).eval()

input_pairs = [
    ("allatonce.wav", "allatonce"),
    ("filtered.wav", "filtered"),
]

for input_file, base_name in input_pairs:
    with sf.SoundFile(input_file, "r") as f:
        input_audio = f.read(44100 * 30, dtype="float32", always_2d=True)
        input_audio = torch.tensor(input_audio.T).reshape(1, 1, -1).cuda()

    for r in [Eleven78.FOUR, Eleven78.EIGHT, Eleven78.TWELVE]:
        for a in range(1, 8):
            for l in range(1, 8):
                parameters = torch.tensor(
                    [
                        [
                            [
                                eleven78_to_0_1(r),
                                attack_to_0_1(a),
                                release_to_0_1(l),
                            ]
                        ]
                    ]
                ).cuda()

                predicted_audio = model(input_audio, parameters)

                sf.write(
                    f"test/{base_name}_{r}_{a}_{l}.wav",
                    predicted_audio.squeeze().detach().cpu().numpy(),
                    44100,
                )
