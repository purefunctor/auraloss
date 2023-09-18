import json
with open('silent_frames.json','r') as rf:
    silence = json.loads(rf.read())
    print(len(silence))

    hop_length = 512
    one_second = 44100//hop_length
    O = []
    for x in range(len(silence)-1):
        if silence[x+1]-silence[x] > one_second:
            O.append((silence[x],silence[x+1]))
    TS = 0
    for x,y in O:
        TS += ((y-x)*512 // 16384)
    print(TS)