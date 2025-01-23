import soundfile as sf
import librosa
import os

from tqdm import tqdm

resample_path = "/home/doyourthing_zyk/zip_file/clean_testset_wav/"
outpath = "/home/doyourthing_zyk/data/vctkdataset/resample_val/"

filelist = os.listdir(resample_path)
targetsr = 16000
for name in tqdm(filelist):
    fullpath = os.path.join(resample_path, name)
    wav, sr = sf.read(fullpath)
    rewav = librosa.resample(wav,orig_sr=sr,target_sr=targetsr)
    outname = os.path.join(outpath,name)
    sf.write(outname, rewav, targetsr)
