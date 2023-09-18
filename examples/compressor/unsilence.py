import json
from silences import hop_length
import os
import subprocess
import boto3

# Initialize the S3 client
s3 = boto3.client('s3')
D = [x for x in os.listdir('mxaudio') if (not ('z_' in x)) and ('o_' in x) & ('.wav' in x) & ('MX20' in x)]
with open('silent_frames.json','r') as rf:
    silence = json.loads(rf.read())

    hop_length = 512
    five_seconds = (44100 * 5)//hop_length
    O = []
    for x in range(len(silence)-1):
        if silence[x+1]-silence[x] > five_seconds:
            O.append((silence[x],silence[x+1]))
    TS = 0
    for x,y in O:
        TS += ((y-x)*512 // 16384)
    for f in os.listdir('./temp'):
        os.remove(os.path.join('./temp', f))
    for FI in D:
        PATH = os.path.join('./mxaudio',FI)
        OPATH = os.path.join('./mxaudio','z_'+FI)
        FIS = []
        for x,y in O:
            XF = os.path.join('./temp',str(x))
            FIS.append(XF+'.wav')
            subprocess.call(f'sox {PATH} {XF}.wav trim {x*hop_length}s {(y-x)*hop_length}s', shell=True)
        subprocess.call(f'sox {" ".join(FIS)} {OPATH}', shell=True)
        with open(OPATH, 'rb') as f:
            s3.upload_fileobj(f, 'meeshkan-datasets', f'mxaudio-unsilenced/{FI}')
        os.remove(OPATH)