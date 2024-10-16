import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EncoderModule(nn.Module):

    def __init__(self, in_channels, nb_filters, kernel_size=20):
        super(EncoderModule, self).__init__()

        k_size = kernel_size
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU(),
            nn.Conv1d(nb_filters, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU()
            #   Keras实现中没有添加激活函数
        )

    def forward(self, x):
        conv_outputs = self.convs(x)
        max_pool_output = F.max_pool1d(conv_outputs, kernel_size=2, stride=2, padding=0)
        return conv_outputs, max_pool_output


class BottomModule(nn.Module):

    def __init__(self, in_channels, nb_filters, kernel_size=20):
        super(BottomModule, self).__init__()

        k_size = kernel_size
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU(),
            nn.Conv1d(nb_filters, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU()
            #   Keras实现中没有添加激活函数
        )

    def forward(self, x):
        conv_outputs = self.convs(x)
        return conv_outputs


class AfterconvModule(nn.Module):

    def __init__(self, in_channels, nb_filters, kernel_size=20):
        super(AfterconvModule, self).__init__()

        k_size = kernel_size
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU()
            #   Keras实现中没有添加激活函数
        )

    def forward(self, x):
        conv_outputs = self.convs(x)
        return conv_outputs


class DecoderModule(nn.Module):

    def __init__(self, in_channels, nb_filters, kernel_size=20):
        super(DecoderModule, self).__init__()

        k_size = kernel_size
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU(),
            nn.Conv1d(nb_filters, nb_filters, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU()
            #   Keras实现中没有添加激活函数
        )

    def forward(self, x):
        conv_outputs = self.convs(x)
        return conv_outputs


class UTimeHead(nn.Module):

    def __init__(self, in_channels, channels):
        super(UTimeHead, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, padding='same', kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels // 4, channels, padding='same', kernel_size=1, bias=False)
        )

    def forward(self, x):
        return F.softmax(self.layers(x), dim=1)


class UTime(nn.Module):

    def __init__(self, input_shape, nb_classes=4, batch_size=64, lr=0.001, nb_filters=32, use_residual=True,
                 use_bottleneck=True, depth=3, kernel_size=80, nb_epochs=1500):

        super(UTime, self).__init__()

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.nb_epochs = nb_epochs
        self.log_transform = LogTransform()
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.name = "UTime"

        down_layers = []
        bottom_layer = []
        up_layers = []
        up_after_layers = []
        up_sample_layers = []
        input_raw = input_shape

        for d in range(self.depth):
            if d == 0:
                down_layers.append(
                    EncoderModule(2 ** d * input_shape[0], 2 ** d * self.nb_filters, self.kernel_size // (2 ** d)))
            else:
                down_layers.append(EncoderModule(2 ** (d - 1) * self.nb_filters, 2 ** d * self.nb_filters,
                                                 self.kernel_size // (2 ** d)))

        for d in range(self.depth):
            up_after_layers.append(
                AfterconvModule(2 ** (self.depth - d) * self.nb_filters, 2 ** (self.depth - d - 1) * self.nb_filters,
                                self.kernel_size // (2 ** (self.depth - d - 1))))

        bottom_layer.append(BottomModule(2 ** (self.depth - 1) * self.nb_filters, 2 ** self.depth * self.nb_filters,
                                         self.kernel_size // (2 ** self.depth)))

        for d in range(self.depth):
            up_layers.append(
                DecoderModule(2 ** (self.depth - d) * self.nb_filters, 2 ** (self.depth - d - 1) * self.nb_filters,
                              self.kernel_size // (2 ** (self.depth - d - 1))))

        for d in range(self.depth):
            up_sample_layers.append(nn.Upsample(size=2 ** d * int(1000 / 2 ** (self.depth - 1)), mode='nearest'))


        self.down_layers = nn.Sequential(*down_layers)
        self.bottom_layer = nn.Sequential(*bottom_layer)
        self.up_after_layers = nn.Sequential(*up_after_layers)
        self.up_layers = nn.Sequential(*up_layers)
        self.up_sample_layers = nn.Sequential(*up_sample_layers)

        self.head = UTimeHead(32, nb_classes)

    def forward(self, x):
        x = self.log_transform(x)
        input_res = []

        for layer in self.down_layers:
            res, x = layer(x)
            input_res.append(res)

        x = self.bottom_layer(x)

        for i in range(self.depth):
            x = self.up_sample_layers[i](x)
            x = self.up_after_layers[i](x)
            x = torch.cat((x, input_res[-1 - i]), dim=1)
            x = self.up_layers[i](x)

        x = self.head(x)

        return x.squeeze(-1)
