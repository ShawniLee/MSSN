import math

from utils import *
import torch.nn.functional as F


class PreConv(nn.Module):
    def __init__(self, input_dim, output_dim, groups):
        super(PreConv, self).__init__()
        self.conv1_1 = nn.Conv1d(input_dim, output_dim, 1, groups=groups)

    def forward(self, x):
        out = self.conv1_1(x)
        return out


class CompressConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, stride=3):
        super(CompressConv, self).__init__()

        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        return out


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


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 device: str,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_q2 = torch.nn.Linear(d_model, q)
        self.W_k2 = torch.nn.Linear(d_model, q)
        self._q = d_model
        self.k = q
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, q, k, v):
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        Q = self.W_q2(Q)
        K = self.W_k2(K)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        score = F.softmax(score, dim=-1)
        _, topk_indices = torch.topk(score, k=self.k, dim=-1)

        # 创建一个全零的矩阵，形状与输入相同
        weights_matrix = torch.zeros_like(score)
        # 对每行的前K个最大值的索引位置设置为1
        weights_matrix.scatter_(-1, topk_indices, 1)
        attention = torch.matmul(weights_matrix, score)
        attention = torch.matmul(score, V)

        return attention


class FeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 device: str,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        # self.downsampling = nn.Conv1d(d_model, d_model, kernel_size = 3, stride = 2)
        self.ffn = FeedForward(d_model, d_hidden)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.atten = MultiHeadAttention(d_model, q, device, dropout)

    def forward(self, x, x_e):
        # x = self.downsampling(x)
        d_encode = self.ffn(x)
        d_encode1 = F.relu(d_encode)
        d_encode2 = self.ln1(d_encode1)

        e_encode = self.ffn(x_e)
        e_encode1 = F.relu(e_encode)
        e_encode2 = self.ln1(e_encode1)

        out = self.atten(e_encode2, e_encode2, d_encode2)
        out = out + d_encode1
        out = self.ln2(out)
        out = out + x

        return out


class AfterSampleConvModule(nn.Module):

    def __init__(self, in_channels, nb_filters, kernel_size=20):
        super(AfterSampleConvModule, self).__init__()

        k_size = kernel_size
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU(),
            nn.Conv1d(nb_filters, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU()
        )

    def forward(self, x):
        conv_outputs = self.convs(x)
        return conv_outputs


class UTMAS(nn.Module):
    def __init__(self, input_shape, nb_classes=2, N=1, d_model=256, d_hidden=256, q=8, device=None):

        super(UTMAS, self).__init__()

        self.N = N
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.q = q
        self.device = device
        self.name = 'UTMAS'

        self.log_transform = LogTransform()
        self.preconv = PreConv(input_shape[0], self.d_model, 1)
        self.position = nn.Conv1d(d_model, d_model, kernel_size=20, padding='same')
        self.encoder_list_1 = nn.ModuleList([Encoder(
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            q=self.q,
            dropout=0.1,
            device=self.device) for _ in range(self.N)])

        self.encoder_list_2 = nn.ModuleList([FeedForward(d_model, d_hidden) for _ in range(self.N)])
        self.encoder_list_3 = nn.ModuleList([FeedForward(d_model, d_hidden) for _ in range(self.N)])

        self.down_sampling_list = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) for _ in range(self.N)])

        self.up_sample = nn.Upsample(scale_factor=2, mode='linear')
        self.up_conv = AfterSampleConvModule(d_model, d_model)
        self.conv_out = nn.Conv1d(self.d_model, nb_classes, 1, padding='same')
        layers = []
        input_raw = input_shape

    def forward(self, x, stage="test"):
        score_list = []
        x = self.log_transform(x)
        x = self.preconv(x)
        encoding_1 = x
        encoding_1 = encoding_1 + self.position(encoding_1)
        encoding_need = []

        encoding_1 = encoding_1.transpose(-1, -2)
        for encoder, downSamplinglayer, fc1, fc2 in zip(self.encoder_list_1, self.down_sampling_list,
                                                        self.encoder_list_2, self.encoder_list_3):
            encoding_1 = fc1(encoding_1)
            encoding_1 = encoding_1.transpose(-1, -2)
            encoding_1 = downSamplinglayer(encoding_1)
            encoding_1 = encoding_1.transpose(-1, -2)
            encoding_need.append(encoding_1)
            encoding_1 = encoder(encoding_1, encoding_1)
            encoding_1 = fc2(encoding_1)

        encoding_need = encoding_need[:-1]
        for encoder, eout in zip(reversed(self.encoder_list_1), reversed(encoding_need)):
            encoding_1 = encoding_1.transpose(-1, -2)
            encoding_1 = self.up_sample(encoding_1)
            encoding_1 = encoding_1.transpose(-1, -2)
            encoding_1 = encoder(encoding_1, encoding_1)
            encoding_1 = fc2(encoding_1)
        encoding_1 = encoding_1.transpose(-1, -2)
        encoding_1 = self.up_sample(encoding_1)
        encoding_1 = encoding_1 + x
        encoding_1 = self.up_conv(encoding_1)
        out = self.conv_out(encoding_1)
        return out
