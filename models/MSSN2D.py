import torch.nn.functional as F
# from utils import *
import torch
import torch.nn as nn


class LogTransform(nn.Module):
    def __init__(self):
        super(LogTransform, self).__init__()
        self.epsilon = nn.Parameter(torch.zeros(1, 1, 20, 1))
        self.instance_norm = nn.InstanceNorm1d(1, affine=False)

    def forward(self, x):
        # epsilon = torch.exp(self.epsilon)
        # # 将输入数据映射到大于零的范围
        # x_min = x.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
        # x = x - x_min + epsilon
        #
        # # 对数变换
        # x_log = torch.log(x)

        # 逐特征Instance标准化
        x_normalized = self.instance_norm(x.squeeze(dim=1)).reshape(x.shape)

        return x_normalized


class InceptionModule(nn.Module):
    def __init__(self, in_channels, nb_filters, use_bottleneck=True, bottleneck_size=32, kernel_size=40,
                 dilation=(1, 1), stride=(2, 1)):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = False
        self.bottleneck_size = bottleneck_size
        self.dilation = dilation

        if use_bottleneck and in_channels > self.bottleneck_size:
            self.use_bottleneck = use_bottleneck
            self.bottleneck = nn.Sequential(
                SeparableConv2d(in_channels, self.bottleneck_size, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(self.bottleneck_size),
                nn.ReLU(inplace=True)
            )
            in_channels = self.bottleneck_size

        kernel_size_s = [(2, kernel_size // (2 ** i)) for i in range(3)]

        self.convs = nn.ModuleList([
            nn.Sequential(
                SeparableConv2d(in_channels, nb_filters, kernel_size=k_size, padding='same', bias=False,
                                padding_mode='replicate'),
                nn.BatchNorm2d(nb_filters),
                nn.ReLU(inplace=True)
            )
            for k_size in kernel_size_s
        ])

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, nb_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(nb_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.use_bottleneck:
            x = self.bottleneck(x)

        conv_outputs = [conv(x) for conv in self.convs]
        max_pool_output = self.conv1x1(F.max_pool2d(x, kernel_size=(1, 3), stride=1, padding=(0, 1)))

        return torch.cat(conv_outputs + [max_pool_output], dim=1)


class FCNHead_Nopooling(nn.Module):
    def __init__(self, in_channels, channels, dim_size=20):
        super(FCNHead_Nopooling, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=(dim_size, 1), bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels // 4, channels, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        return self.layers(x)
        # return F.sigmoid(self.layers(x))


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False, padding_mode='replicate'):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels,
                                   bias=bias, padding_mode=padding_mode)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FCNHead_Separable(nn.Module):
    def __init__(self, in_channels, channels, dim_size=20):
        super(FCNHead_Separable, self).__init__()

        self.layers = nn.Sequential(
            SeparableConv2d(in_channels, in_channels // 4, kernel_size=(dim_size, 1), padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            SeparableConv2d(in_channels // 4, channels, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        return F.softmax(self.layers(x), dim=1)


class FCNHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(FCNHead, self).__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 1), padding='same', bias=False,
                      padding_mode='replicate'),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels // 4, channels, kernel_size=(1, 1), padding='same', bias=False,
                      padding_mode='replicate')
        )

    def forward(self, x):
        return self.layers(x)
        # return F.sigmoid(self.layers(x))


class MSSN2D(nn.Module):
    def __init__(self, input_shape, nb_classes=2, batch_size=64, lr=0.001, nb_filters=32,
                 use_bottleneck=True, depth=4, kernel_size=80, nb_epochs=1500, dilation=(1, 2), stride=(2, 1)):

        super(MSSN2D, self).__init__()

        self.nb_filters = nb_filters
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.nb_epochs = nb_epochs

        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.log_transform = LogTransform()
        self.name = f"MSSN2D"

        layers = []
        for d in range(self.depth):
            layers.append(InceptionModule(input_shape[0], self.nb_filters, use_bottleneck=self.use_bottleneck,
                                          kernel_size=self.kernel_size, dilation=dilation, stride=stride))
            input_shape = (self.nb_filters * 4, input_shape[1], input_shape[2])
        self.layers = nn.Sequential(*layers)

        self.head = FCNHead_Separable(input_shape[0], nb_classes, dim_size=input_shape[1])

    def forward(self, x):
        x = self.log_transform(x)
        outputs = []
        for layer in self.layers:
            x = layer(x)
            out = self.head(x)
            outputs.append(out.squeeze().unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # 在循环外进行拼接
        return outputs
