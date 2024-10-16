import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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

        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShortcutLayer, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        return F.relu(x1 + x2)


# class FCNHead_Nopooling(nn.Module):
#     def __init__(self, in_channels, channels):
#         super(FCNHead_Nopooling, self).__init__()

#         self.layers = nn.Sequential(
#             nn.Conv1d(in_channels, in_channels // 4, kernel_size=1, bias=False),
#             nn.BatchNorm1d(in_channels // 4),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv1d(in_channels // 4, channels, kernel_size=1, padding=0, bias=False)
#         )

#     def forward(self, x):
#         return F.softmax(self.layers(x), dim=1)

    
class FCNHead_Nopooling(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=1):
        super(FCNHead_Nopooling, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size = kernel_size, bias=False, padding="same"),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels // 4, channels, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        return F.softmax(self.layers(x), dim=1)

class FCNHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(FCNHead, self).__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d((1, None)),
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels // 4, channels, kernel_size=1, padding='same', bias=False)
        )

    def forward(self, x):
        return F.softmax(self.layers(x), dim=1)


class Transformer_lognorm_association(nn.Module):
    def __init__(self, input_shape, nb_classes=2, N=1, d_hidden=256, q=8, v=8, h=8, head_kernal_size=1, mode=0, lr=0.001, d_model = 256, d_association=64, nb_epochs=1500):

        super(Transformer_lognorm_association, self).__init__()
        
        self.nb_epochs = nb_epochs
        self.N = N
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.d_association = d_association
        self.q = q
        self.v = v
        self.h = h
        self.mode = mode
        self.head_kernal_size = head_kernal_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bottleneck_size = 16
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.name = "Sleep_Transformer"
        
        self.log_transform = LogTransform()
        self.embedding_channel = torch.nn.Linear(input_shape[0], self.d_model)
        self.position = nn.Conv1d(d_model, d_model, kernel_size=20, padding='same')
        
        self.encoder_list_1 = nn.ModuleList([Encoder(d_model=self.d_model,
                                                  d_hidden=self.d_hidden,
                                                  q=self.q,
                                                  v=self.v,
                                                  h=self.h,
                                                  mask=False,
                                                  dropout=0.1,
                                                  device=self.device) for _ in range(self.N)])
        
        self.embedding_association = torch.nn.Linear(input_shape[1] * h, self.d_association)
        
        self.conv_embedding_association = nn.Sequential(
            torch.nn.Conv2d(in_channels=h,out_channels=self.d_association,kernel_size=(1,40),stride=(1,3),padding='valid'),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_association, self.d_association, kernel_size=[1,10], stride=(1,3), padding='valid'),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=(1, 3), stride=(1,3))
        )
        self.conv_embedding_linear = torch.nn.Linear(self.d_association, 1)
        
        
        
        layers = []
        input_raw = input_shape
       
        self.head_mode0 = FCNHead_Nopooling(self.d_association, nb_classes, self.head_kernal_size)
        self.head_mode1 = FCNHead_Nopooling(104 , nb_classes, self.head_kernal_size)
        self.head_mode2 = FCNHead_Nopooling(self.d_model + 104 , nb_classes, self.head_kernal_size)

    def forward(self, x, stage="test"):
        x = self.log_transform(x)
        # x B,C,L 
        encoding_1 = x.transpose(-1, -2)
        encoding_1 = self.embedding_channel(encoding_1)
        encoding_1 = encoding_1.transpose(-1, -2)
        encoding_1 = encoding_1 + self.position(encoding_1)
        encoding_1 = encoding_1.transpose(-1, -2)
        score_list = []
        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)
            score_list.append(score_input)
            
        # #线性层提取特征
        # #只用最后一层的联系矩阵
        # score_input = torch.cat(score_input.chunk(self.h, dim=0), dim=-1)  # score_input (b, l, l * h)
        # association_feature = self.embedding_association(score_input)  # association_feature (b, l, d_association * h)
        
        #卷积层提取特征
        score_input = torch.stack(score_input.chunk(self.h, dim=0), dim=1)  # score_input (b, h, l, l)
        association_feature = self.conv_embedding_association(score_input)  # association_feature (b, l, d_association * h)
        association_feature = association_feature.permute(0, 2, 3, 1)
        association_feature = self.conv_embedding_linear(association_feature)
        association_feature = association_feature.squeeze()
        
        
        #score_list.append(association_feature)
        #association_feature = torch.cat(score_list,dim=-1)
        encoding_1 = encoding_1.transpose(-1, -2)
        association_feature = association_feature.transpose(-1, -2)
        
        #x = torch.cat([encoding_1,x], dim=1)
        if self.mode == 0:
            result = self.head_mode0(encoding_1)
        elif self.mode == 1:
            result = self.head_mode1(association_feature)
        elif self.mode == 2:
            if (len(association_feature.shape)==2):
                association_feature = association_feature.unsqueeze(0)
            concatenated_tensor = torch.cat((encoding_1, association_feature),dim=1)
            result = self.head_mode2(concatenated_tensor)
        return result.squeeze(), score_list
    
  