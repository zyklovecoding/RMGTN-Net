import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
import os
from functools import partial
from geomloss import SamplesLoss
import soundfile as sf
import librosa

class PerceptualLoss(nn.Module):
    def __init__(self, model_type='wav2vec', PRETRAINED_MODEL_PATH = '/path/to/wav2vec_large.pt'):
        super().__init__()
        self.model_type = model_type
        self.wass_dist = SamplesLoss()
        if model_type == 'wav2vec':
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([PRETRAINED_MODEL_PATH])
            self.model = model[0]
            self.model = self.model.feature_extractor
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('Please assign a loss model')
            sys.exit()

    def forward(self, y_hat, y):
        y_hat, y = map(self.model, [y_hat, y])
        return self.wass_dist(y_hat, y)
        # for PFPL-W-MAE or PFPL-W
        # return torch.abs(y_hat - y).mean()

# # 加载语音数据（假设语音文件为 mono，采样率为 16kHz）
# audio_file = r"/home/doyourthing_zyk/clarity/data/clarity_CEC2_data/clarity_data/dev/scenes/S06833_target_anechoic_CH1.wav" # 替换为你的语音文件路径
# waveform, sample_rate = sf.read(audio_file)
#
# waveform = librosa.resample(waveform.T, orig_sr=sample_rate, target_sr=16000)
# # 将语音数据处理为模型输入格式
# waveform = torch.from_numpy(waveform).to(torch.float32)
#
# p = PerceptualLoss(PRETRAINED_MODEL_PATH="/home/doyourthing_zyk/claity_focus/McNet/src/W2CLOSS/modelwights/wav2vec_large.pt")
#
# loss = p(waveform,waveform)
# print(loss)