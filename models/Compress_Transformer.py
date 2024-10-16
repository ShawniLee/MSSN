import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
import math

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
        x_normalized = self.instance_norm(x_log[:,:,:]).reshape(x_log.shape)

        return x_normalized



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score

    
    
    
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

    
class PreConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PreConv, self).__init__()
        self.conv1_1 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        out = self.conv1_1(x)
        return out
    
class CompressConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size = 5, stride = 3):
        super(CompressConv, self).__init__()

        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size = kernel_size, stride = stride)

    def forward(self, x):
        out = self.conv(x)
        return out
    
    
class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage):
        #实验五
        x = x.transpose(-1, -2)
        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)
        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)
        x = x.transpose(-1, -2)
        return x, score

class Compress_Transformer(nn.Module):
    def __init__(self, input_shape, nb_classes=2, N=1, d_hidden=256, q=8, v=8, h=8, head_kernal_size=1, compressKernelSize=5, compressStride=3, lr=0.001, d_model = 256, nb_epochs=1500):

        super(Compress_Transformer, self).__init__()
        
        self.nb_epochs = nb_epochs
        self.N = N
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.q = q
        self.v = v
        self.h = h
        self.head_kernal_size = head_kernal_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.name = 'Compress_Transformer'
        self.compressKernelSize = compressKernelSize
        self.compressStride = compressStride
        #self.log_transform = LogTransform()
                 
        self.conv_out = nn.Conv1d(self.d_model, nb_classes, self.head_kernal_size, padding='same')
                 
        self.position = nn.Conv1d(d_model, d_model, kernel_size=20, padding='same')
        self.preconv = PreConv(input_shape[0], self.d_model)
        self.compressConv = CompressConv(self.d_model, self.d_model, self.compressKernelSize, self.compressStride)
        self.encoder_list_1 = nn.ModuleList([Encoder(
                                                  d_model=self.d_model,
                                                  d_hidden=self.d_hidden,
                                                  q=self.q,
                                                  v=self.v,
                                                  h=self.h,
                                                  mask=False,
                                                  dropout=0.1,
                                                  device=self.device) for _ in range(self.N)])
        
        self.up_sample = nn.Upsample(size = input_shape[1], mode='nearest')
        self.up_conv = AfterSampleConvModule(d_model, d_model)
        
        layers = []
        input_raw = input_shape
       

    def forward(self, x, stage="test"):
        score_list = []
    
        x = self.preconv(x)
        encoding_1 = self.compressConv(x)
        encoding_1 = encoding_1 + self.position(encoding_1)
        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)
            score_list.append(score_input)
        # print(encoding_1.shape)
        encoding_1 = self.up_sample(encoding_1)
        encoding_1 = encoding_1 + x
        encoding_1 = self.up_conv(encoding_1)
        out = self.conv_out(encoding_1)
        return out
        
        
  