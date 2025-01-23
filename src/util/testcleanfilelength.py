import os

import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

filepath = "/home/doyourthing_zyk/data/vctkdataset/resample_train/"
filelist = os.listdir(filepath)

minval = 100000000
maxval = -1
a = list()
co = 0
dd = ""
for name in tqdm(filelist):
    filep = os.path.join(filepath,name)
    wav,sr = sf.read(filep)
    audio = AudioSegment.from_wav(filep)
    # Calculate the number of segments
    length_ms = len(audio)
    if length_ms<2000:
        co +=1
    a.append(length_ms)
    if length_ms<minval:
        minval = length_ms
    if length_ms>maxval:
        maxval = length_ms
        dd = filep

print(f"最小{minval}")
print(max(a))
print(sum(a)/len(a))
print(co)
print(dd)