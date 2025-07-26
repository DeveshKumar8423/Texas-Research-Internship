import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    
    # --- ADDED SAFEGUARD ---
    # Ensure k is not larger than the number of available frequencies
    num_frequencies = frequency_list.shape[0]
    k = min(k, num_frequencies -1)
    if k == 0: k = 1 # Must have at least one frequency

    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.k = configs['top_k']
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=configs['d_model'], out_channels=configs['d_ff'], kernel_size=(1, configs['num_kernels']), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=configs['d_ff'], out_channels=configs['d_model'], kernel_size=(1, configs['num_kernels']), padding='same'),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res

class TimesNet(nn.Module):
    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.embedding = nn.Linear(configs['num_features'], configs['d_model'])
        self.layers = nn.ModuleList([TimesBlock(configs) for _ in range(configs['e_layers'])])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(configs['seq_len'] * configs['d_model'], configs['num_classes'])
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        return self.fc(x)