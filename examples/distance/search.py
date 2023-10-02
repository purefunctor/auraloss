from itertools import product
from tcn import TCNModule
from torchsummary import summary

N_BLOCKS = range(4, 11)
DILATION_GROWTH = [2, 3, 5, 10]
KERNEL_SIZE = [5, 13, 15, 20, 25]
CHANNEL_WIDTH = [32, 64, 96]

configurations = []
for nblocks, dilation_growth, kernel_size, channel_width in product(
    N_BLOCKS, DILATION_GROWTH, KERNEL_SIZE, CHANNEL_WIDTH
):
    model = TCNModule(
        nblocks=nblocks,
        kernel_size=kernel_size,
        dilation_growth=dilation_growth,
        channel_width=channel_width,
        stack_size=nblocks,
    )
    info = summary(model, verbose=0)
    trainable_params = info.trainable_params
    receptive_field = model.compute_receptive_field() / 44100 * 1000

    if 200_000 > trainable_params > 100_000 and 100 < receptive_field < 1000:
        print(
            f"{nblocks=},{dilation_growth=},{kernel_size=},{channel_width=},{trainable_params},{receptive_field}ms"
        )
        configurations.append(
            {
                "nblocks": nblocks,
                "dilation_growth": dilation_growth,
                "kernel_size": kernel_size,
                "channel_width": channel_width,
                "trainable_params": trainable_params,
            }
        )

print(f"Total: {len(configurations)}")


with open("train.sh", "w") as f:
    f.write("#!/usr/bin/bash")
    f.write("\n\n")
    for configuration in sorted(configurations, key=lambda configuration: configuration["trainable_params"]):
        f.write(
            "python train.py --nblocks={nblocks} --dilation_growth={dilation_growth} --kernel_size={kernel_size} --channel_width={channel_width}  # {trainable_params}".format(
                **configuration
            )
        )
        f.write("\n")
