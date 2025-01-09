import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import statistics
from collections import Counter

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = 128 # configs.hidden_dim
        self.num_layers = 4 # configs.num_layers

        # Decompsition Kernel Size
        kernel_size = 7 #configs.moving_avg
        self.decompsition = series_decomp(kernel_size)
        self.individual = False # configs.individual
        self.channels = configs.enc_in

        self.lstm_seasonal = nn.LSTM(15, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.2)
        self.linear_seasonal = nn.Linear(self.hidden_dim, self.pred_len)
        self.lstm_trend = nn.LSTM(15, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.2)
        self.linear_trend = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        #seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # import pdb
        # pdb.set_trace()
        out, _ = self.lstm_seasonal(seasonal_init)
        seasonal_output = self.linear_seasonal(out[:, -1, :])
        out, _ = self.lstm_trend(trend_init)
        trend_output = self.linear_trend(out[:, -1, :])

        x = seasonal_output + trend_output
        return x.unsqueeze(2) # to [Batch, Output length, Channel]