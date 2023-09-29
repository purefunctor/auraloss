import librosa
import numpy as np

# Compute the short-time amplitude envelope using RMS energy
frame_length = 1024
hop_length = 512


def get_silent_frames(infi):
    # Load the audio file
    y, sr = librosa.load(infi, sr=None)

    rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Identify silent regions (frames)
    silent_frames0_01 = np.where(rmse < 0.01)[1]
    silent_frames0_001 = np.where(rmse < 0.001)[1]
    silent_frames0_0001 = np.where(rmse < 0.0001)[1]
    return silent_frames0_0001.tolist()
