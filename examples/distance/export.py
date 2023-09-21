from tcn import TCNModule
import torch
import wandb


if __name__ == "__main__":
    api = wandb.Api()

    artifact = api.artifact("meeshkan/distance-near-to-far/model-92cbh8ab:v19")
    weights = artifact.get_path("model.ckpt").download("/tmp")
    model = TCNModule.load_from_checkpoint(weights).eval()
    x = torch.rand((1, 1, 44100 // 4))
    model.to_onnx(
        "micro-tcn-static.onnx",
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
