from tcn import TCNModule
import torch
import wandb

api = wandb.Api()

artifact = api.artifact("meeshkan/near-to-far/model-824yyzni:v19")
weights = artifact.get_path("model.ckpt").download("/tmp")
model = TCNModule.load_from_checkpoint(weights).eval()

for s in [512, 1024, 2048]:
    x = torch.rand((1, 1, s + model.compute_receptive_field()))
    model.to_onnx(
        f"micro-tcn-300-{s}-{model.compute_receptive_field()}.onnx",
        x,
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes={
        #     "input": {
        #         0: "batch_size",
        #         2: "samples",
        #     },
        #     "output": {
        #         0: "batch_size",
        #         2: "samples",
        #     },
        # },
    )
