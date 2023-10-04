from pathlib import Path
from tcn import TCNModule
import torch
import wandb

_exports = Path("exports")
if not _exports.exists():
    _exports.mkdir()

api = wandb.Api()

for run in api.runs(path="meeshkan/near-to-far"):
    if len(run.tags) > 0 or run.state != "finished":
        continue
    artifact = api.artifact(f"meeshkan/near-to-far/model-{run.id}:latest")
    weights = artifact.get_path("model.ckpt").download("/tmp")
    model = TCNModule.load_from_checkpoint(
        weights, map_location=torch.device("cpu")
    ).eval()
    r = model.compute_receptive_field()

    for s in [2, 4, 6, 8, 16]:
        x = torch.rand((1, 1, 512 * s + r))
        model.to_onnx(
            f"exports/{run.name}-{512 * s}.onnx",
            x,
            input_names=["input"],
            output_names=["output"],
        )
