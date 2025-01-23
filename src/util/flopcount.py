import torch
from torch import nn
from thop import profile
import time
from torch.autograd import profiler
from fvcore.nn import FlopCountAnalysis
from torch.utils.flop_counter import FlopCounterMode
import os
import numpy as np
import soundfile as sf

class RNN_FC(nn.Module):

    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            act_funcs=('SiLU', ''),
            use_FC=True,
    ):
        super(RNN_FC, self).__init__()

        # Sequence layer
        self.sequence_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # If batch_first is true, the shape of input tensor should be [batch_size,time_step,feature]. The output is the same.
            bidirectional=bidirectional,
        )
        self.sequence_model.flatten_parameters()

        # Fully connected layer
        self.use_FC = use_FC
        if self.use_FC:
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        self.act_funcs = []
        for act_func in act_funcs:
            if act_func == 'SiLU' or act_func == 'swish':
                self.act_funcs.append(nn.SiLU())
            elif act_func == 'ReLU':
                self.act_funcs.append(nn.ReLU())
            elif act_func == 'Tanh':
                self.act_funcs.append(nn.Tanh())
            elif act_func == None or act_func == '':
                self.act_funcs.append(None)  # type:ignore
            else:
                raise NotImplementedError(f"Not implemented activation function {act_func}")

    def forward(self, x):
        """
        Args:
            x: [B, T, Feature]
        Returns:
            [B, T, Feature]
        """
        o, _ = self.sequence_model(x)
        if self.act_funcs[0] is not None:
            o = self.act_funcs[0](o)
        if self.use_FC:
            o = self.fc_output_layer(o)
            if self.act_funcs[1] is not None:
                o = self.act_funcs[1](o)
        return o
class McNet(nn.Module):
    def __init__(
            self,
            freq=None,
            narr=None,
            sub=None,
            full=None,
            order=['freq', 'narr', 'sub', 'full'],
            sub_freqs=(3, 2),
            look_past_and_ahead=(5, 0),
            learning_rate=0.001,
            optimizer_kwargs=dict(),
            lr_scheduler='ExponentialLR',
            lr_scheduler_kwargs={'gamma': 0.992},
            exp_name="exp",
            use_dense_net=False,
            use_time_domain_loss=False,
    ):
        super(McNet, self).__init__()
        self.freq = freq if not isinstance(freq, list) else nn.Sequential(*freq)
        self.narr = narr if not isinstance(narr, list) else nn.Sequential(*narr)
        self.sub = sub if not isinstance(sub, list) else nn.Sequential(*sub)
        self.full = full if not isinstance(full, list) else nn.Sequential(*full)
        self.look_past_and_ahead = look_past_and_ahead
        self.sub_freqs = sub_freqs
        # self.io = io
        self.order = order
        self.use_dense_net = use_dense_net
        self.use_time_domain_loss = use_time_domain_loss

    def prepare_input(self, x, *args, **kwargs):

        X = torch.view_as_complex(x.view(1, 6, 208, 256, 2)).permute(0, 2, 3, 1)  # (batch, time frame, freq, channel)
        time = 49152

        # normalization by using ref_channel
        batch_size, frame_num, freq_num, chn_num = X.shape
        Xr = X[..., 4].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=(1, 2)).reshape(batch_size, 1, 1,
                                                      1)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X

        X /= (XrMM + 1e-8)
        input = torch.view_as_real(X).reshape(batch_size, frame_num, freq_num, chn_num * 2)
        XrMag = torch.abs(X[:, :, :, 4]).unsqueeze(-1)
        XMag = torch.abs(X)
        return {'input': input, "X": X, "XrMM": XrMM, "XrMag": XrMag, "XMag": XMag, "original_time_len": time}

    def forward(self, x):
        # x:(B,C,T,F)
        data = self.prepare_input(x)
        i = data['input']
        B, T, F, H = i.shape
        index = 1
        input_dict = {}

        for curr in self.order:
            if self.use_dense_net:
                for idx in range(2, index):
                    i = torch.concat([i, input_dict[idx - 1]], dim=-1)
                B, T, F, H = i.shape
            if not curr.startswith('sub2') and not curr.startswith('sub3') and not curr.startswith('full4freq'):
                if curr.endswith('+X'):
                    i = torch.concat([i, data['input']], dim=-1)
                    curr = curr.replace('+X', '')
                elif curr.endswith('+XrMag'):
                    i = torch.concat([i, data['XrMag']], dim=-1)
                    curr = curr.replace('+XrMag', '')

            reduce_by_num_freqs = False
            if curr == 'sub_':
                reduce_by_num_freqs = True
                curr = 'sub'

            if curr == 'freq':
                i = i.reshape(B * T, F, -1)
                i = self.freq(i).reshape(B, T, F, -1)
            elif curr == 'narr':
                i = i.permute(0, 2, 1, 3).reshape(B * F, T, -1)
                i = self.narr(i)
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)
            elif curr == 'sub':
                i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                i = torch.concat([i[:, :, :self.sub_freqs, :], i, i[:, :, -self.sub_freqs:, :]], dim=2)
                i = torch.nn.functional.unfold(i, kernel_size=(self.sub_freqs * 2 + 1, 1))
                i = i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1)
                if reduce_by_num_freqs:
                    i = i / (self.sub_freqs * 2 + 1)
                i = self.sub(i)
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)
            elif curr.startswith('sub2'):
                if curr.endswith('+X'):
                    X = data['input'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                else:
                    assert curr.endswith('+XrMag'), curr
                    X = data['XrMag'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                if self.sub_freqs != 0:
                    X = torch.concat([X[:, :, :self.sub_freqs, :], X, X[:, :, -self.sub_freqs:, :]], dim=2)
                    Xsub = torch.nn.functional.unfold(X, kernel_size=(self.sub_freqs * 2 + 1, 1))
                    i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F)
                else:
                    Xsub = X.reshape(B * T, -1, F)
                    i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F)
                i = torch.concat([i, Xsub], dim=1)
                i = i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1)
                i = self.sub(i)
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)
            elif curr.startswith('sub3'):
                assert len(self.sub_freqs) == 2
                if curr.endswith('+X'):
                    X = data['input'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                else:
                    assert curr.endswith('+XrMag'), curr
                    X = data['XrMag'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                if self.sub_freqs[0] != 0:
                    X = torch.concat([X[:, :, :self.sub_freqs[0], :], X, X[:, :, -self.sub_freqs[0]:, :]], dim=2)
                    Xsub = torch.nn.functional.unfold(X, kernel_size=(self.sub_freqs[0] * 2 + 1, 1))
                else:
                    Xsub = X.reshape(B * T, -1, F)
                i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                i = torch.concat([i[:, :, :self.sub_freqs[1], :], i, i[:, :, -self.sub_freqs[1]:, :]], dim=2)
                i = torch.nn.functional.unfold(i, kernel_size=(self.sub_freqs[1] * 2 + 1, 1))
                i = torch.concat([i, Xsub], dim=1)
                i = i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1)
                i = self.sub(i)
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)
            elif curr.startswith('full2'):
                i = data['XMag'].permute(0, 3, 1, 2)
                B, H, T, F = i.shape
                i = i.reshape(B * H, T, F)
                i = self.full(i)
                i = i.reshape(B, H, T, F, -1).permute(0, 2, 3, 1, 4)
                i = i.reshape(B, T, F, -1)
            elif curr.startswith('full3freq'):
                if curr.endswith('-allch'):
                    i = data['XMag'].permute(0, 2, 3, 1)
                    B, F, H, T = i.shape
                    i = torch.nn.functional.pad(i, pad=self.look_past_and_ahead, mode='constant', value=0)
                    i = i.reshape(B * F * H, 1, -1, 1)
                    i = torch.nn.functional.unfold(i, kernel_size=(
                    self.look_past_and_ahead[0] + self.look_past_and_ahead[1] + 1, 1))
                    i = i.reshape(B, F, H, -1, T).permute(0, 2, 4, 1, 3).reshape(B * H * T, F, -1)
                    i = self.full(i)
                    i = i.reshape(B, H, T, F, -1).permute(0, 2, 3, 1, 4)
                    i = i.reshape(B, T, F, -1)
                else:
                    i = data['XrMag'].permute(0, 2, 3, 1)
                    B, F, H, T = i.shape
                    i = torch.nn.functional.pad(i, pad=self.look_past_and_ahead, mode='constant', value=0)
                    i = i.reshape(B * F, H, -1, 1)
                    i = torch.nn.functional.unfold(i, kernel_size=(
                    self.look_past_and_ahead[0] + self.look_past_and_ahead[1] + 1, 1))
                    i = i.reshape(B, F, -1, T).permute(0, 3, 1, 2).reshape(B * T, F, -1)
                    i = self.full(i).reshape(B, T, F, -1)
            elif curr.startswith('fullfreq'):
                i = i.reshape(B * T, F, -1)
                i = self.full(i).reshape(B, T, F, -1)
            elif curr.startswith('full4freq'):
                i = i.reshape(B * T, F, -1)
                if curr.endswith('+XrMag'):
                    XrMag = data['XrMag'].permute(0, 2, 3, 1)
                    XrMag = torch.nn.functional.pad(XrMag, pad=self.look_past_and_ahead, mode='constant', value=0)
                    XrMag = XrMag.reshape(B * F, 1, -1, 1)
                    XrMag = torch.nn.functional.unfold(XrMag, kernel_size=(
                    self.look_past_and_ahead[0] + self.look_past_and_ahead[1] + 1, 1))
                    XrMag = XrMag.reshape(B, F, -1, T).permute(0, 3, 1, 2).reshape(B * T, F, -1)
                    i = torch.cat([i, XrMag], dim=-1)
                i = self.full(i).reshape(B, T, F, -1)
            else:
                assert curr == 'full', curr
                i = i.reshape(B, T, -1)
                i = self.full(i)
                i = i.reshape(B, T, F, -1)
            input_dict[index] = i
            index += 1

        # data['output'] = i
        # prediction = self.io.prepare_prediction(**data)
        prediction = i
        return prediction

def profile_model(model, input_size, device='cpu'):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Dummy input based on the input size
    inputs = torch.randn(1, *input_size).to(device)

    # Start the profiler
    with profiler.profile(use_cuda=(device == 'cuda')) as prof:
        with profiler.record_function("model_inference"):
            outputs = model(inputs)

    # Print the profiler results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def test_model_speed(model, input_size, device='cpu', num_tests=100):
    # Generate a dummy wav file
    sample_rate = 16000  # Typical sample rate
    num_channels = input_size[0]  # Number of channels

    # Calculate the number of samples corresponding to the ISTFT of the frames
    n_fft = 510
    hop_length = 255
    num_frames = input_size[1]
    num_samples = n_fft + hop_length * (num_frames - 1)

    audio_signal = np.random.randn(num_samples, num_channels).astype(np.float32)
    wav_filename = 'dummy_audio.wav'
    sf.write(wav_filename, audio_signal, sample_rate)

    model = model.to(device)
    model.eval()

    # Warm up the device if using CUDA
    if device == 'cuda':
        dummy_input = torch.randn(1, *input_size).to(device)
        for _ in range(50):
            _ = model(dummy_input)

    # Start timing
    start_time = time.time()
    for _ in range(num_tests):
        # Load the wav file
        audio_data, sr = sf.read(wav_filename, dtype='float32')
        audio_data = audio_data.T  # Shape: (channels, samples)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(device)  # Shape: (1, channels, samples)

        # Apply STFT
        stft_output = torch.stft(audio_tensor, n_fft=510, hop_length=255, return_complex=True)
        inputs = torch.abs(stft_output)

        with torch.no_grad():
            outputs = model(inputs)
    end_time = time.time()

    # Calculate average inference time
    average_time = (end_time - start_time) / num_tests
    print(
        f"Average inference time per forward pass (including data loading and preprocessing): {average_time:.6f} seconds")

    # Clean up
    os.remove(wav_filename)


def calculate_macs_and_paras(model, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create a dummy input based on the input size
    inputs = torch.randn(1, *input_size).to(device)

    # Calculate MACs using thop
    macs, paras = profile(model, inputs=(inputs,), verbose=False)

    # Initialize fvcore's FlopCountAnalysis
    # flops_analyzer = FlopCountAnalysis(model, inputs)

    # Get the total FLOPs from fvcore
    macs = macs / 1e9  # Convert to MMACs for better readability

    # Convert MACs to human-readable format (usually MACs are very large numbers)
    paras = paras / 1e9  # Convert to M

    print(f"MACs: {macs:.6f} MMACs")
    print(f"params: {paras:.6f} M")

    return macs, paras


def calculate_flops(model, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create a dummy input based on the input size
    inputs = torch.randn(1, *input_size).to(device)

    # Use the FlopCounterMode to calculate FLOPs
    with FlopCounterMode(model) as flop_counter:
        _ = model(inputs)  # Execute the model to trigger the flop counting

    # Get the total FLOPs from the FlopCounterMode
    total_flops = flop_counter.get_total_flops() / 1e9  # Convert to GFLOPs for better readability
    print(f"FLOPs: {total_flops:.6f} GFLOPs")

    return total_flops


if __name__ == "__main__":
    # Define the models
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # test_model_speed(net, (1,16000), device='cpu', num_tests=20)
    # profile_model(net, (1,16000))

    mcnet = McNet(
        freq=RNN_FC(input_size=6 * 2, output_size=64, hidden_size=128, num_layers=1, bidirectional=True,
                    act_funcs=('', 'ReLU'), use_FC=True),
        narr=RNN_FC(input_size=76, output_size=64, hidden_size=256, num_layers=1, bidirectional=False,
                    act_funcs=('', 'ReLU'), use_FC=True),
        sub=RNN_FC(input_size=327, output_size=64, hidden_size=384, num_layers=1, bidirectional=False,
                   act_funcs=('', 'ReLU'), use_FC=True),
        full=RNN_FC(input_size=70, output_size=2, hidden_size=128, num_layers=1, bidirectional=True, act_funcs=('', ''),
                    use_FC=True),
        order=['freq', 'narr+X', 'sub3+XrMag', 'full4freq+XrMag'],
        sub_freqs=(3, 2),
        look_past_and_ahead=(5, 0),
        # io=io,
    )

    model = mcnet
    inputs = torch.randn(12, 208, 256)  # batch_size=1, channels=3, height=32, width=32

    # 使用 thop 来统计 FLOPs 和参数量
    flops, params = profile(model, inputs=(inputs,))

    # 打印结果
    flops_g = flops / 1e9  # 转换为 GFLOPs
    params_m = params / 1e6  # 转换为 M (百万) 参数
    print(flops_g)
    print(params_m)