from tcn import TCNModule
import torch


if __name__ == "__main__":
    model = TCNModule(kernel_size=15, channel_width=32, dilation_growth=2, lr=0.001)
    x = torch.rand((1, 1, 44100))
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
