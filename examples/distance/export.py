from tcn import TCNModule
import torch
import wandb

api = wandb.Api()

artifacts = [
    "meeshkan/near-to-far/model-w2sg3k9w:v19",
    "meeshkan/near-to-far/model-sdl0oecr:v19",
    "meeshkan/near-to-far/model-t02dv6ru:v19",
    "meeshkan/near-to-far/model-qj6nboda:v19",
]

for artifact in artifacts:
    artifact = api.artifact(artifact)
    weights = artifact.get_path("model.ckpt").download("/tmp")
    model = TCNModule.load_from_checkpoint(weights).eval()
    r = model.compute_receptive_field()

    for s in [512, 1024, 2048]:
        x = torch.rand((1, 1, s + r))
        model.to_onnx(
            f"exports/{artifact.logged_by().name}-{s}.onnx",
            x,
            input_names=["input"],
            output_names=["output"],
        )
