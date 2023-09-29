import json
from silences import hop_length, get_silent_frames
import os
import subprocess
import sys

D = [
    x
    for x in os.listdir("mxaudio")
    if (not ("z_" in x)) and ("o_x_" in x) & (".wav" in x) & ("MX20" in x)
]
print(D)
for FI in D:
    IPATH = os.path.join("./mxaudio", FI)
    silence = get_silent_frames(IPATH)

    hop_length = 512
    five_seconds = (44100 * 5) // hop_length
    O = []
    for x in range(len(silence) - 1):
        if silence[x + 1] - silence[x] > five_seconds:
            O.append((silence[x], silence[x + 1]))
    for FI2 in [FI, FI.replace("_x_", "_y_")]:
        for f in os.listdir("./temp"):
            os.remove(os.path.join("./temp", f))
        FIS = []
        OPATH = os.path.join("./mxaudio", "z_" + FI2)
        PATH = os.path.join("./mxaudio", FI2)

        for x, y in O:
            XF = os.path.join("./temp", str(x))
            FIS.append(XF + ".wav")
            subprocess.call(
                f"sox {PATH} {XF}.wav trim {x*hop_length}s {(y-x)*hop_length}s",
                shell=True,
            )
        subprocess.call(f'sox {" ".join(FIS)} {OPATH}', shell=True)
    for FI2 in [FI, FI.replace("_x_", "_y_")]:
        OPATH = os.path.join("./mxaudio", "z_" + FI2)
        print(f"Uplaoding {FI2}")
        subprocess.call(
            f"aws s3 cp {OPATH} s3://meeshkan-datasets/mxaudio-unsilenced/{FI2}",
            shell=True,
        )
        os.remove(OPATH)
