import torch
import torch.nn as nn
import math


class ECA(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output



class ECAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, gamma=2, b=1):
        super(ECAB, self).__init__()

        self.ecab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ECA(num_feat, gamma, b)
            )
        # self.conv = nn.Conv2d(num_feat, num_feat, (3, 1), 1, (1, 0))
        # self.conv2 = nn.Conv2d(num_feat, num_feat, (1, 3), 1, (0, 1))

    def forward(self, x):
        x1 = self.ecab(x)
        return x1


class ECAB2(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, gamma=2, b=1):
        super(ECAB2, self).__init__()

        self.conv = nn.Conv2d(num_feat, num_feat // compress_ratio, (3, 1), 1, (1, 0))
        self.conv2 = nn.Conv2d(num_feat, num_feat // compress_ratio, (1, 3), 1, (0, 1))
        self.relu = nn.GELU()
        self.conv3 = nn.Conv2d(num_feat // compress_ratio, num_feat, (3, 1), 1, (1, 0))
        self.conv4 = nn.Conv2d(num_feat // compress_ratio, num_feat, (1, 3), 1, (0, 1))
        self.eca = ECA(num_feat, gamma, b)

    def forward(self, x):
        x1 = self.conv(x) + self.conv2(x)
        x2 = self.relu(x1)
        x3 =self.conv3(x2) + self.conv4(x2)
        x4 = self.eca(x3)
        return x4


class ECAB3(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, gamma=2, b=1):
        super(ECAB3, self).__init__()

        self.conv = nn.Conv2d(num_feat, num_feat // compress_ratio, (3, 1), 1, (1, 0))
        self.conv2 = nn.Conv2d(num_feat, num_feat // compress_ratio, (1, 3), 1, (0, 1))
        self.relu = nn.GELU()
        self.conv3 = nn.Conv2d(num_feat // compress_ratio * 2, num_feat, 3, 1, 1)
        # self.conv4 = nn.Conv2d(num_feat // compress_ratio, num_feat, (1, 3), 1, (0, 1))
        self.eca = ECA(num_feat, gamma, b)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv2(x)
        x_f = torch.concat([x1, x2], dim=1)
        x3 = self.relu(x_f)
        x4 = self.conv3(x3)
        x5 = self.eca(x4)
        return x5

class Convlayer(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, gamma=2, b=1):
        super(Convlayer, self).__init__()

        self.ecab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            # ECA(num_feat, gamma, b)
            )
        # self.conv = nn.Conv2d(num_feat, num_feat, (3, 1), 1, (1, 0))
        # self.conv2 = nn.Conv2d(num_feat, num_feat, (1, 3), 1, (0, 1))

    def forward(self, x):
        x1 = self.ecab(x)
        return x1

if __name__ == '__main__':
    import numpy as np
    x1 = torch.from_numpy(np.ones((2, 64, 128, 128))).float()
    model = ECAB3(
        num_feat=64, compress_ratio=3, gamma=2, b=1)
    x = model.forward(x1)
    print(x.shape)