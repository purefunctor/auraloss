import os
import subprocess

I = []
O = []
for DIR in ["./data/day1_unsilenced", "./data/day2_unsilenced"]:
    A = os.listdir(DIR)
    I += [f"{DIR}/{x}" for x in A if "nt1_middle" in x]
    O += [f"{DIR}/{x}" for x in A if "67_near" in x]

for x, y in zip(I, O):
    assert x.split("_")[-1] == y.split("_")[-1]
    assert "nt" in x
    assert "67" in y

subprocess.call(f'sox {" ".join(I)} nt.wav', shell=True)
subprocess.call(f'sox {" ".join(O)} 67.wav', shell=True)
