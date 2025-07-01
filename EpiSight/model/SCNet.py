import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class SpatialChannelConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class SCCBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1 ,downsample=None):
        super(SCCBasicBlock, self).__init__()
        self.conv1 = SpatialChannelConv(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpatialChannelConv(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SelfCalibratedConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SelfCalibratedConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm2d):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SelfCalibratedConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        # out = self.conv3(torch.cat([out_a, out_b], dim=1))
        # out = self.bn3(out)
        #
        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #
        # out += residual
        # out = self.relu(out)
        out = torch.cat([out_a, out_b], dim=1)

        return out


class CNNNet(nn.Module):

    def __init__(self, w):
        super(CNNNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.1)

        self.scbottleneck = SCBottleneck(128, 128)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.1)

        self.sccbasicblock = SCCBasicBlock(64, 64)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.1)

        self.fc = nn.Linear(w*4*32, 128)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.scbottleneck(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.sccbasicblock(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


class BiLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        # self.embed_smi = nn.Embedding(size, hidden)
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1, bidirectional=True)
        self.lin1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        # self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.squeeze(1)
        # embed = self.embed_smi(data)  # (N,L,128)
        out, (h_n, _) = self.bilstm(x)
        # out = self.pool(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.lin1(out)
        # out = self.pool(out)
        out = torch.mean(out, 1)

        return out


class SCC(nn.Module):
    def __init__(self, window, input_size, hidden_size, num_layers):
        super(SCC, self).__init__()
        self.conv = CNNNet(window)
        self.bilstm = BiLstm(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bilstm(x)
        out = torch.cat((x1, x2), dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        return out
