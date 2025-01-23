import os
import soundfile as sf
import torch
from torchmetrics.functional.audio import perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import short_time_objective_intelligibility as stoi
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from tqdm import tqdm

modelname = ["mcnet","mc2split","eabnet","cepnet"]
for modelna in modelname:
    cleandir = "/home/doyourthing_zyk       /data/finaldataset/val/clean/"
    noisydir = "/home/doyourthing_zyk/predictwav/"+modelna

    cleanlist = [os.path.join(cleandir,a) for a in os.listdir(cleandir)]
    cleanlist.sort()
    noisylist = [os.path.join(noisydir,a) for a in os.listdir(noisydir)]
    noisylist.sort()

    sum_SDR = 0.0
    sum_WB_PESQ = 0.0
    sum_SISDR = 0.0
    sum_NB_PESQ = 0.0
    sum_STOI = 0.0
    count = 0

    for yr_hat,yr in tqdm(zip(noisylist,cleanlist)):
        yr_hat_t = sf.read(yr_hat, dtype='float32')[0]
        yr_t = sf.read(yr, dtype='float32')[0]
        yr_hat_t = torch.from_numpy(yr_hat_t)
        yr_t = torch.from_numpy(yr_t)
        SDR = sdr(preds=yr_hat_t, target=yr_t)
        WB_PESQ = pesq(preds=yr_hat_t, target=yr_t, fs=16000, mode='wb')
        SISDR = si_sdr(preds=yr_hat_t, target=yr_t)
        NB_PESQ = pesq(preds=yr_hat_t, target=yr_t, fs=16000, mode='nb')
        STOI = stoi(preds=yr_hat_t, target=yr_t, fs=16000)
        # Accumulate the metrics
        sum_SDR += SDR
        sum_WB_PESQ += WB_PESQ
        sum_SISDR += SISDR
        sum_NB_PESQ += NB_PESQ
        sum_STOI += STOI
        count += 1
    average_SDR = sum_SDR / count if count else 0
    average_WB_PESQ = sum_WB_PESQ / count if count else 0
    average_SISDR = sum_SISDR / count if count else 0
    average_NB_PESQ = sum_NB_PESQ / count if count else 0
    average_STOI = sum_STOI / count if count else 0

    filepath = f"/home/doyourthing_zyk/predictwav/{modelna}.txt"
    # Print the average metrics
    # 打开文件（如果文件不存在则创建文件，若存在则覆盖）
    with open(filepath, "w") as file:
        file.write(f'Average SDR: {average_SDR:.4f}\n')
        file.write(f'Average WB_PESQ: {average_WB_PESQ:.4f}\n')
        file.write(f'Average SISDR: {average_SISDR:.4f}\n')
        file.write(f'Average NB_PESQ: {average_NB_PESQ:.4f}\n')
        file.write(f'Average STOI: {average_STOI:.4f}\n')

    print("Results have been written to 'results.txt'")
