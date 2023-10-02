from itertools import islice, product, zip_longest
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


configurations.sort(key=lambda configuration: configuration["trainable_params"])
configuration_groups = [
    list(islice(configurations, i, i + 4)) for i in range(0, len(configurations), 4)
]

with open("train.sh", "w") as f:
    f.write("#!/usr/bin/bash")
    for configuration_group in zip_longest(*configuration_groups):
        f.write("\n\n")
        for configuration in configuration_group:
            if configuration is None:
                continue
            f.write(
                "python train.py --nblocks={nblocks} --dilation_growth={dilation_growth} --kernel_size={kernel_size} --channel_width={channel_width} --half  # {trainable_params}\n".format(
                    **configuration
                )
            )
