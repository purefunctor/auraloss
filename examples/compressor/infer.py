import librosa
import soundfile as sf
from tcn import TCNModel
import torch

# FILENAME = "o_x_MX20.TWO.wav"
# FRAME_LENGTH = 2048
# HOP_LENGTH = 512
# NUM_SECONDS_OF_SLICE = 2

# sound, sr = librosa.load(FILENAME, sr=None)

# clip_rms = librosa.feature.rms(
#     y=sound, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
# )

# clip_rms = clip_rms.squeeze()
# peak_rms_index = clip_rms.argmax()
# peak_index = peak_rms_index * HOP_LENGTH + int(FRAME_LENGTH / 2)

peak_index = 578278400

model = TCNModel.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=11-step=7008.ckpt").eval().cuda().half()

input_sound, _ = librosa.load("o_x_MX20.TWO.wav", sr=44100, offset=peak_index // 44100 - 3, duration=10)
sf.write("input.wav", input_sound, samplerate=44100)

target_sound, _ = librosa.load("o_y_MX20.TWO.wav", sr=44100, offset=peak_index // 44100 - 3, duration=10)
sf.write("target.wav", target_sound, samplerate=44100)

input_sound = torch.tensor(input_sound).cuda().half().unsqueeze(0).unsqueeze(0)
parameters = torch.tensor([1.0, 0.2]).cuda().half().unsqueeze(0).unsqueeze(0)
predicted_sound = model(input_sound, parameters)

predicted_sound = predicted_sound.float().squeeze().detach().cpu().numpy()
sf.write("prediction.wav", predicted_sound, samplerate=44100)
