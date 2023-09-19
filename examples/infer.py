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
    near_to_far = DistanceModel.load_from_checkpoint(
        "./weights/near-to-far/model.ckpt"
    ).eval()

    print("Loading far-to-near.ckpt")
    far_to_near = DistanceModel.load_from_checkpoint(
        "./weights/far-to-near/model.ckpt"
    ).eval()

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

    overrides = {
        "414_near_Hellekka_90006112_96893396.wav": (2036892, 5272757),
        "414_near_Hanna_176930767_186600560.wav": (3162927, 6373489),
        "414_near_Hanna_258244414_265166192.wav": (1316161, 4555416),
        "414_near_Aku_371016506_379433021.wav": (0, 2918831),
        "414_near_Aku_347865343_356919995.wav": (0, 3354236),
    }

    for index, (key, near_file) in enumerate(near_files.items()):
        print(f"[{index + 1}/{len(near_files)}] {near_file.stem}")
        far_file = far_files[key]

        bucket_path = output_files / "distance" / "{}_{}_{}".format(*key)
        if not bucket_path.exists():
            bucket_path.mkdir(parents=True)

        if near_file.name in overrides:
            start, stop = overrides[near_file.name]
            near_audio, _ = sf.read(
                near_file, always_2d=True, dtype="float32", start=start, stop=stop
            )
            far_audio, _ = sf.read(
                far_file, always_2d=True, dtype="float32", start=start, stop=stop
            )
        else:
            near_audio, _ = sf.read(near_file, always_2d=True, dtype="float32")
            far_audio, _ = sf.read(far_file, always_2d=True, dtype="float32")

        sf.write(bucket_path / "near.wav", near_audio, 44100)
        sf.write(bucket_path / "far.wav", far_audio, 44100)

        near_audio = torch.tensor(near_audio.T).unsqueeze(0).cuda()
        far_audio = torch.tensor(far_audio.T).unsqueeze(0).cuda()

        to_far = near_to_far(near_audio).squeeze().detach().cpu().numpy()
        sf.write(bucket_path / "to_far.wav", to_far, 44100)

        to_near = far_to_near(far_audio).squeeze().detach().cpu().numpy()
        sf.write(bucket_path / "to_near.wav", to_near, 44100)


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
    compressor_bn_artifact.get_path("model.ckpt").download("./weights/compressor-bn/")

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

    input_files = []
    pattern = re.compile(r"^67_near_(\w+)_(\d+)_(\d+).wav$")
    for audio_file in audio_files.iterdir():
        match = pattern.match(audio_file.name)
        if match is None:
            continue
        (singer, start, end) = match.groups()
        input_files.append((audio_file, singer, start, end))

    overrides = {
        "67_near_Hellekka_90006112_96893396.wav": (2036892, 5272757),
        "67_near_Hanna_176930767_186600560.wav": (3162927, 6373489),
        "67_near_Hanna_258244414_265166192.wav": (1316161, 4555416),
        "67_near_Aku_371016506_379433021.wav": (0, 2918831),
        "67_near_Aku_347865343_356919995.wav": (0, 3354236),
    }

    for index, (input_file, singer, start, end) in enumerate(input_files):
        if input_file.stem in [
            "67_near_Hanna_189061126_194545658",
            "67_near_Aku_387688563_392218764",
        ]:
            print("Skipping, too large")
            continue
        bucket_path = output_files / "compressor" / f"{singer}_{start}_{end}"
        print(f"[{index + 1}/{len(input_files)}] {input_file.stem}")
        if not bucket_path.exists():
            bucket_path.mkdir(parents=True)

        if input_file.name in overrides:
            start, stop = overrides[input_file.name]
            input_audio, _ = sf.read(
                input_file, always_2d=True, dtype="float32", start=start, stop=stop
            )
        else:
            input_audio, _ = sf.read(input_file, always_2d=True, dtype="float32")

        input_audio = torch.tensor(input_audio.T).unsqueeze(0).cuda()

        for value in [0.2, 0.4, 0.6, 0.8]:
            parameter = torch.tensor([[[value]]]).cuda()

            multi = (
                compressor_multi(input_audio, parameter)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            sf.write(bucket_path / f"multi_{int(value * 10)}.wav", multi, 44100)

            unsilenced = (
                compressor_unsilenced(input_audio, parameter)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            sf.write(
                bucket_path / f"unsilenced_{int(value * 10)}.wav", unsilenced, 44100
            )

        single = compressor_bn(input_audio).squeeze().detach().cpu().numpy()
        sf.write(bucket_path / f"single_4.wav", single, 44100)


def upscale_main():
    print("Downloading upscale weights.")
    nt1_to_u67_artifact = api.artifact(
        "meeshkan/nt1-to-u67-auraloss/model-75esyg20:v19"
    )
    nt1_to_u67_artifact.get_path("model.ckpt").download(
        "./weights/nt1-to-u67-auraloss/"
    )

    nt1_to_u67 = UpscaleModel.load_from_checkpoint(
        "./weights/nt1-to-u67-auraloss/model.ckpt"
    ).eval()
    print("Testing upscale models.")
    x = torch.rand((1, 1, 44100)).cuda()
    print("nt1-to-u67.ckpt", nt1_to_u67(x).shape)

    input_files = {}
    target_files = {}
    pattern = re.compile(r"^(nt1|67)_(\w+)_(\w+)_(\d+)_(\d+).wav$")
    for audio_file in audio_files.iterdir():
        match = pattern.match(audio_file.name)
        if match is None:
            continue
        (name, distance, singer, start, end) = match.groups()
        if (name, distance) == ("nt1", "middle"):
            input_files[(singer, start, end)] = audio_file
        elif (name, distance) == ("67", "near"):
            target_files[(singer, start, end)] = audio_file

    for index, (key, input_file) in enumerate(input_files.items()):
        print(f"[{index + 1}/{len(input_files)}] {input_file.stem}")
        target_file = target_files[key]

        bucket_path = output_files / "upscale" / "{}_{}_{}".format(*key)
        if not bucket_path.exists():
            bucket_path.mkdir(parents=True)

        overrides = {
            "nt1_middle_Hellekka_90006112_96893396.wav": (2036892, 5272757),
            "nt1_middle_Hanna_176930767_186600560.wav": (3162927, 6373489),
            "nt1_middle_Hanna_258244414_265166192.wav": (1316161, 4555416),
            "nt1_middle_Aku_371016506_379433021.wav": (0, 2918831),
            "nt1_middle_Aku_347865343_356919995.wav": (0, 3354236),
        }

        if input_file.name in overrides:
            start, stop = overrides[input_file.name]
            input_audio, _ = sf.read(
                input_file, always_2d=True, dtype="float32", start=start, stop=stop
            )
            target_audio, _ = sf.read(
                target_file, always_2d=True, dtype="float32", start=start, stop=stop
            )
        else:
            input_audio, _ = sf.read(input_file, always_2d=True, dtype="float32")
            target_audio, _ = sf.read(target_file, always_2d=True, dtype="float32")

        sf.write(bucket_path / "input.wav", input_audio, 44100)
        sf.write(bucket_path / "target.wav", target_audio, 44100)

        input_audio = torch.tensor(input_audio.T).unsqueeze(0).cuda()
        to_far = nt1_to_u67(input_audio).squeeze().detach().cpu().numpy()
        sf.write(bucket_path / "upscaled.wav", to_far, 44100)


if __name__ == "__main__":
    # distance_main()
    # compressor_main()
    upscale_main()
