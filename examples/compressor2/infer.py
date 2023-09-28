import soundfile as sf
from data import Eleven78, eleven78_to_0_1, attack_to_0_1, release_to_0_1
import librosa
from tcn import TCNModel
import torch
import wandb

api = wandb.Api()

artifact = api.artifact("meeshkan/eleven78-compressor-unsilenced/model-yeasc3a8:v19")
weights = artifact.get_path("model.ckpt").download("/tmp/")
model = TCNModel.load_from_checkpoint(weights).eval()

input_pairs = [
    (
        "67_near_day1_ratio4_attack7_release5_Aku_542232920_579457368.wav",
        "67_1178_2_day1_ratio4_attack7_release5_Aku_542232920_579457368.wav",
        Eleven78.FOUR,
        7,
        5,
    ),
    # (
    #     "67_near_day1_ratio8_attack3_release7_Hellekka_18653024_52546144.wav",
    #     "67_1178_2_day1_ratio8_attack3_release7_Hellekka_18653024_52546144.wav",
    #     Eleven78.EIGHT,
    #     3,
    #     7,
    # ),
    # (
    #     "67_near_day2_ratio12_attack1_release3_Amanda_443023360_457268493.wav",
    #     "67_1178_2_day2_ratio12_attack1_release3_Amanda_443023360_457268493.wav",
    #     Eleven78.TWELVE,
    #     1,
    #     3,
    # ),
]

# FRAME_LENGTH = 2048
# HOP_LENGTH = 512
# NUM_SECONDS_OF_SLICE = 2

# peak_indices = []
# for (input_file, *_) in input_pairs:
#     y, sr = librosa.load(input_file, sr=None)
#     clip_rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
#     clip_rms = clip_rms.squeeze()
#     peak_rms_index = clip_rms.argmax()
#     peak_index = peak_rms_index * HOP_LENGTH + int(FRAME_LENGTH / NUM_SECONDS_OF_SLICE)
#     peak_indices.append(peak_index)

peak_indices = [
    32165376,
    # 3876864,
    # 12015104,
]

for index, ((input_file, target_file, ratio, attack, release), peak_index) in enumerate(
    zip(input_pairs, peak_indices)
):
    with sf.SoundFile(input_file, "r") as f:
        f.seek(peak_index - 44100 * 5)
        input_audio = f.read(44100 * 10, dtype="float32", always_2d=True)
        sf.write(f"sweep/input_{index}.wav", input_audio, 44100)

    with sf.SoundFile(target_file, "r") as f:
        f.seek(peak_index - 44100 * 5)
        target_audio = f.read(44100 * 10, dtype="float32", always_2d=True)
        sf.write(f"sweep/target_{index}.wav", target_audio, 44100)

    input_audio = torch.tensor(input_audio.T).reshape(1, 1, -1).cuda()
    target_audio = torch.tensor(target_audio.T).reshape(1, 1, -1).cuda()

    for r in [Eleven78.FOUR, Eleven78.EIGHT, Eleven78.TWELVE]:
        for a in range(1, 8):
            for l in range(1, 8):
                parameters = torch.tensor(
                    [
                        [
                            [
                                eleven78_to_0_1(r),
                                attack_to_0_1(a),
                                release_to_0_1(l),
                            ]
                        ]
                    ]
                ).cuda()

                predicted_audio = model(input_audio, parameters)

                rn = f"ratioCanon{r}" if r == ratio else f"ratio{r}"
                an = f"attackCanon{a}" if a == attack else f"attack{a}"
                ln = f"releaseCanon{l}" if l == release else f"release{l}"

                sf.write(
                    f"sweep/predicted_{index}_{rn}_{an}_{ln}.wav",
                    predicted_audio.squeeze().detach().cpu().numpy(),
                    44100,
                )
