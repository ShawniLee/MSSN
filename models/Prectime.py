import torch.nn.functional as F
from einops import rearrange

from utils import *


class LogTransform(nn.Module):
    def __init__(self, epsilon=1):
        super(LogTransform, self).__init__()
        self.epsilon = epsilon
        self.instance_norm = nn.InstanceNorm1d(20, affine=False)

    def forward(self, x):
        # 将输入数据映射到大于零的范围
        x_min = x.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
        x = x - x_min + self.epsilon

        # 对数变换
        x_log = torch.log(x)

        # 逐特征Instance标准化
        x_normalized = self.instance_norm(x_log[:, :, :]).reshape(x_log.shape)

        return x_normalized


class IWFE(nn.Module):
    def __init__(self, in_channels):
        super(IWFE, self).__init__()
        self.conv_embedding1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=5, stride=1, dilation=1, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding='same'),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding='same'),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding='same'),
        )

        self.conv_embedding2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=5, stride=1, dilation=4, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, dilation=4, padding='same'),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, dilation=4, padding='same'),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, dilation=4, padding='same'),
        )

    def forward(self, x):
        f1 = self.conv_embedding1(x)
        f2 = self.conv_embedding2(x)
        f = torch.concat((f1, f2), dim=1)
        f = rearrange(f, 'b c l -> b l c')
        flatten_f = torch.flatten(f, 1)
        return flatten_f, f


# (B,c,l)
# (1,c,l) -> (B,c,l) -> (1, c, l)
class IWCD(nn.Module):
    def __init__(self):
        super(IWCD, self).__init__()
        self.rnn1 = nn.LSTM(6400, 100, 2, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(200, 100, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        b = x.shape[0]
        # h0, c0 = torch.randn(4,b,100) ,torch.randn(4,b,100)
        # h1, c1 = torch.randn(4,b,200) ,torch.randn(4,b,200)
        # x, _ = self.rnn1(x, (h0,c0))
        # x, _ = self.rnn2(x, (h1,c1))
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return x


class IWPR(nn.Module):
    def __init__(self, n):
        super(IWPR, self).__init__()

        self.conv_embedding = nn.Sequential(
            nn.Conv1d(in_channels=400, out_channels=128, kernel_size=5, stride=1, dilation=1, padding='same'),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding='same'),
            nn.Dropout(p=0.5),
        )
        self.full_connected = nn.Linear(128, n)

    def forward(self, f):
        f = self.conv_embedding(f)
        f = self.full_connected(f.transpose(1, 2))
        return f


class Prectime(nn.Module):
    def __init__(self, in_channels, n_class):
        super(Prectime, self).__init__()
        self.iwfe = IWFE(in_channels)
        self.iwcd = IWCD()
        self.iwpr = IWPR(n_class)
        self.upSample = nn.Upsample(scale_factor=18, mode='linear')
        self.upSample2 = nn.Upsample(scale_factor=50, mode='linear')
        self.full_connected = nn.Linear(200, n_class)
        self.log_transform = LogTransform()

        self.name = "Prectime"

    def forward(self, x):
        x = self.log_transform(x)
        x = torch.chunk(x, 20, dim=-1)
        x = torch.stack(x, dim=1)
        # print(x.shape)
        flatten_f = []
        f = []
        # (b,n,c,l)
        for i in range(x.shape[1]):
            _flatten_f, _f = self.iwfe(x[:, i, :, :])
            flatten_f.append(_flatten_f)
            f.append(_f)
        flatten_f = torch.stack(flatten_f, dim=1)
        f = torch.stack(f, dim=1)

        d = self.iwcd(flatten_f)
        o1 = d
        output1 = self.full_connected(o1)
        output1 = rearrange(output1, 'b l c -> b c l')

        output1 = self.upSample2(output1)
        d = self.upSample(d)
        d = rearrange(d, 'b n (l t) -> b n l t', l=25)
        # d = rearrange(d, 'b n l c -> b n c l')

        y = torch.cat([d, f], dim=3)
        y = rearrange(y, 'b n l c -> b n c l')

        output = []
        # y = torch.cat([d, f],dim=2)
        for i in range(y.shape[1]):
            _output = self.iwpr(y[:, i, :, :])
            output.append(_output)
        output2 = torch.cat(output, dim=1)
        # output2 = F.softmax(output2, dim=2)
        # x = self.iwcd(x)
        # x = self.iwpr(x)
        # return output1, output2
        return output2.permute(0, 2, 1), output1
