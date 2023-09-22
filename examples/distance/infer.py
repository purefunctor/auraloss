import soundfile as sf
from tcn import TCNModule
import torch
import wandb


api = wandb.Api()

artifacts = [
    ("TCN-100", "meeshkan/near-to-far/model-o4p6cz29:v19"),
    ("TCN-300", "meeshkan/near-to-far/model-824yyzni:v19"),
]

for model_name, artifact_name in artifacts:
    artifact = api.artifact(artifact_name)
    weights = artifact.get_path("model.ckpt").download(f"/tmp/{model_name}")
    model = TCNModule.load_from_checkpoint(
        weights, map_location=torch.device("cpu")
    ).eval()
    receptive_field = model.compute_receptive_field()

    with sf.SoundFile("data/day1_unsilenced/414_near_far_close_30_1192.wav", "r") as f:
        f.seek(88 * 44100 - receptive_field)
        input_audio = f.read(44100 * 10 + receptive_field, dtype="float32")

    with sf.SoundFile("data/day1_unsilenced/414_far_far_far_65_1192.wav", "r") as f:
        f.seek(88 * 44100)
        target_audio = f.read(44100 * 10, dtype="float32")

    i = torch.tensor(input_audio).unsqueeze(0).unsqueeze(0)
    y = model(i).squeeze().float().detach().numpy()

    sf.write(
        f"414_near_{model_name}.wav", input_audio[receptive_field:], samplerate=44100
    )
    sf.write(f"414_far_{model_name}.wav", target_audio, samplerate=44100)
    sf.write(f"414_pred_{model_name}.wav", y, samplerate=44100)
