import librosa
import numpy as np

# Load the audio file
y, sr = librosa.load('./mxaudio/o_x_MX20.EIGHT.wav', sr=None)

# Compute the short-time amplitude envelope using RMS energy
frame_length = 1024
hop_length = 512

if __name__ == '__main__':
    rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Identify silent regions (frames)
    silent_frames0_01 = np.where(rmse < 0.01)[1]
    silent_frames0_001 = np.where(rmse < 0.001)[1]
    silent_frames0_0001 = np.where(rmse < 0.0001)[1]

    #print(len(silent_frames0_01)/len(rmse[1]))
    #print(len(silent_frames0_001)/len(rmse[1]))
    import json
    with open('silent_frames.json','w') as wf:
        wf.write(json.dumps(silent_frames0_0001.tolist()))