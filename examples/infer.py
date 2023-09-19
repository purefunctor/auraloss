from distance.tcn import TCNModule as DistanceModel
from compressor.tcn import TCNModel as CompressorModel
from upscale.tcn import TCNModule as UpscaleModel
from pathlib import Path
import re
import soundfile as sf
import torch
import wandb


api = wandb.Api()
audio_files = Path.cwd() / "data"
output_files = Path.cwd() / "out"


def distance_main():
    print("Downloading distance weights.")
    far_to_near_artifact = api.artifact(
        "meeshkan/distance-far-to-near/model-71jxvdc9:v39"
    )
    far_to_near_artifact.get_path("model.ckpt").download("./weights/far-to-near/")

    print("Loading near-to-far.ckpt")
    near_to_far = DistanceModel.load_from_checkpoint("./weights/near-to-far/model.ckpt").eval()

    print("Loading far-to-near.ckpt")
    far_to_near = DistanceModel.load_from_checkpoint("./weights/far-to-near/model.ckpt").eval()

    print("Testing distance models.")
    x = torch.rand((1, 1, 44100)).cuda()
    print("near-to-far.ckpt", near_to_far(x).shape)
    print("far-to-near.ckpt", far_to_near(x).shape)
    
    near_files = {}
    far_files = {}
    pattern = re.compile(r"^414_(\w+)_(\w+)_(\d+)_(\d+).wav$")
    for audio_file in audio_files.iterdir():
        match = pattern.match(audio_file.name)
        if match is None:
            continue
        (distance, singer, start, end) = match.groups()
        if distance == "near":
            near_files[(singer, start, end)] = audio_file
        elif distance == "far":
            far_files[(singer, start, end)] = audio_file

    for index, (key, near_file) in enumerate(near_files.items()):
        print(f"Processed {index + 1}/{len(near_files)}")
        far_file = far_files[key]

        near_audio, _ = sf.read(near_file, always_2d=True, dtype="float32")
        near_audio = torch.tensor(near_audio.T).unsqueeze(0).cuda()

        far_audio, _ = sf.read(far_file, always_2d=True, dtype="float32")
        far_audio = torch.tensor(far_audio.T).unsqueeze(0).cuda()

        to_far = near_to_far(near_audio).squeeze().detach().cpu().numpy()
        to_far_file = output_files / f"414_near_to_far_{key[0]}_{key[1]}_{key[2]}.wav"
        sf.write(to_far_file, to_far, 44100)

        to_near = far_to_near(far_audio).squeeze().detach().cpu().numpy()
        to_near_file = output_files / f"414_far_to_near_{key[0]}_{key[1]}_{key[2]}.wav"
        sf.write(to_near_file, to_near, 44100)


def compressor_main():
    print("Downloading compressor weights.")
    compressor_multi_artifact = api.artifact(
        "meeshkan/unified-compressor/model-nny3uwb6:v49"
    )
    compressor_multi_artifact.get_path("model.ckpt").download(
        "./weights/compressor-multi/"
    )

    compressor_unsilenced_artifact = api.artifact(
        "meeshkan/unified-compressor-unsilenced/model-yj0njbvq:v24"
    )
    compressor_unsilenced_artifact.get_path("model.ckpt").download(
        "./weights/compressor-unsilenced/"
    )

    compressor_bn_artifact = api.artifact(
        "meeshkan/unified-compressor-unsilenced-four/model-p1scllfo:v39"
    )
    compressor_bn_artifact.get_path("model.ckpt").download(
        "./weights/compressor-bn/"
    )

    print("Loading compressor-multi.ckpt")
    compressor_multi = CompressorModel.load_from_checkpoint(
        "./weights/compressor-multi/model.ckpt"
    ).eval()

    print("Loading compressor-unsilenced.ckpt")
    compressor_unsilenced = CompressorModel.load_from_checkpoint(
        "./weights/compressor-unsilenced/model.ckpt"
    ).eval()

    print("Loading compressor-bn.ckpt")
    compressor_bn = CompressorModel.load_from_checkpoint(
        "./weights/compressor-bn/model.ckpt"
    ).eval()

    print("Testing compressor models.")
    x = torch.rand((1, 1, 44100)).cuda()
    p = torch.tensor([[[0.2]]]).cuda()
    print("compressor-multi.ckpt", compressor_multi(x, p).shape)
    print("compressor-unsilenced.ckpt", compressor_unsilenced(x, p).shape)
    print("compressor-bn", compressor_bn(x).shape)


def upscale_main():
    print("Downloading upscale weights.")
    nt1_to_u67_artifact = api.artifact(
        "meeshkan/nt1-to-u67-auraloss/model-75esyg20:v19"
    )
    nt1_to_u67_artifact.get_path("model.ckpt").download(
        "./weights/nt1-to-u67-auraloss/"
    )

    nt1_to_u67 = UpscaleModel.load_from_checkpoint("./weights/nt1-to-u67-auraloss/model.ckpt").eval()
    print("Testing upscale models.")
    x = torch.rand((1, 1, 44100)).cuda()
    print("nt1-to-u67.ckpt", nt1_to_u67(x).shape)


if __name__ == "__main__":
    distance_main()
    # compressor_main()
    # upscale_main()
