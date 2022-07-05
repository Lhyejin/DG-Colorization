import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from skimage import color
import numpy.random as npr
import torch.nn.functional as F
from torchsummary import summary
import random
import cv2


class DownConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out_nc)

    def forward(self, x):
        if self.activation:
            x = self.conv(self.lrelu(x))
        else:
            x = self.conv(x)

        if self.batch_norm:
            return self.bn(x)
        else:
            return x


class UpConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(UpConv, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_nc, out_nc, kernel_size, stride, padding)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(0.5)
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bn = nn.BatchNorm2d(out_nc)

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(self.upconv(self.relu(x)))
        else:
            x = self.upconv(self.relu(x))

        if self.dropout:
            return self.drop(x)
        else:
            return x


class Encoder(nn.Module):
    def __init__(self, in_nc=1, num_filter=64):
        super(Encoder, self).__init__()
        self.down1 = DownConv(in_nc, num_filter, activation=False, batch_norm=False)
        self.down2 = DownConv(num_filter, num_filter * 2)
        self.down3 = DownConv(num_filter * 2, num_filter * 4)
        self.down4 = DownConv(num_filter * 4, num_filter * 8)
        self.down5 = DownConv(num_filter * 8, num_filter * 8)
        self.down6 = DownConv(num_filter * 8, num_filter * 8)
        self.down7 = DownConv(num_filter * 8, num_filter * 8)
        self.down8 = DownConv(num_filter * 8, num_filter * 8, batch_norm=False)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)

        return (x1, x2, x3, x4, x5, x6, x7, x8)

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, DownConv):
                nn.init.normal_(m.conv.weight, mean, std)
            if isinstance(m, UpConv):
                nn.init.normal_(m.upconv.weight, mean, std)


class HeadEncoder(Encoder):
    def __init__(self, in_nc=1, num_filter=64):
        super(HeadEncoder, self).__init__(in_nc=in_nc, num_filter=num_filter)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2) # [-1, 256, 32, 32]

        return [x1, x2, x3]


class TailEncoder(Encoder):
    def __init__(self, num_filter=64):
        super(TailEncoder, self).__init__(num_filter=num_filter)

    def forward(self, x3):
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        return [x4, x5, x6, x7, x8]



class Decoder(nn.Module):
    def __init__(self, num_filter=64, out_nc=2):
        super(Decoder, self).__init__()
        # UpSampling
        self.up1 = UpConv(num_filter * 8, num_filter * 8, dropout=True)
        self.up2 = UpConv(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.up3 = UpConv(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.up4 = UpConv(num_filter * 8 * 2, num_filter * 8)
        self.up5 = UpConv(num_filter * 8 * 2, num_filter * 4)
        self.up6 = UpConv(num_filter * 4 * 2, num_filter * 2)
        self.up7 = UpConv(num_filter * 2 * 2, num_filter)
        self.up8 = UpConv(num_filter * 2, out_nc, batch_norm=False)

    def forward(self, x, adain_layer_idx=5):
        # adain_layer_idx : index of up layer to apply adain 
        x1, x2, x3, x4, x5, x6, x7, x8 = x

        curr_idx = 1
        x_out = self.up1(x8)
        x7 = self._adain(x7, x_out) if curr_idx == adain_layer_idx else x7
        x_out = torch.cat([x_out, x7], 1)
        x_out = self.up2(x_out)

        curr_idx +=1 # 2
        x6 = self._adain(x6, x_out) if curr_idx == adain_layer_idx else x6
        x_out = torch.cat([x_out, x6], 1)
        x_out = self.up3(x_out)

        curr_idx +=1 # 3
        x5 = self._adain(x5, x_out) if curr_idx == adain_layer_idx else x5
        x_out = torch.cat([x_out, x5], 1) 
        x_out = self.up4(x_out)

        curr_idx +=1 # 4
        x4 = self._adain(x4, x_out) if curr_idx == adain_layer_idx else x4
        x_out = torch.cat([x_out, x4], 1)
        x_out = self.up5(x_out)

        curr_idx +=1 # 5
        x3 = self._adain(x3, x_out) if curr_idx == adain_layer_idx else x3
        x_out = torch.cat([x_out, x3], 1)
        x_out = self.up6(x_out)

        curr_idx +=1 # 6
        x2 = self._adain(x2, x_out) if curr_idx == adain_layer_idx else x2
        x_out = torch.cat([x_out, x2], 1)
        x_out = self.up7(x_out)

        curr_idx +=1 # 7
        x1 = self._adain(x1, x_out) if curr_idx == adain_layer_idx else x1
        x_out = torch.cat([x_out, x1], 1)
        x_out = self.up8(x_out)
        x_out = nn.Tanh()(x_out)

        return x_out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, DownConv):
                nn.init.normal_(m.conv.weight, mean, std)
            if isinstance(m, UpConv):
                nn.init.normal_(m.upconv.weight, mean, std)

    def _adain(self, input_z, style_z, eps=1e-5):
        content_mean = input_z.mean(dim=(2,3), keepdim=True)
        content_var = input_z.var(dim=(2,3), keepdim=True)
        input_z = (input_z - content_mean) / (content_var + eps).sqrt()

        style_mean = style_z.mean(dim=(2,3), keepdim=True)
        style_var = style_z.var(dim=(2,3), keepdim=True)
        
        input_z = input_z * (style_var + eps).sqrt() + style_mean
        
        return input_z


class Discriminator(nn.Module):
    def __init__(self, in_nc, num_filter, out_nc):
        super(Discriminator, self).__init__()

        self.conv1 = DownConv(in_nc, num_filter, activation=False, batch_norm=False)
        self.conv2 = DownConv(num_filter, num_filter * 2)
        self.conv3 = DownConv(num_filter * 2, num_filter * 4)
        self.conv4 = DownConv(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = DownConv(num_filter * 8, out_nc, stride=1, batch_norm=False)

    def forward(self, x):
        # x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x_out = self.conv5(x)

        return x_out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, DownConv):
                torch.nn.init.normal_(m.conv.weight, mean, std)


class StyleDiscriminator(Encoder):
    def __init__(self, num_filter=64, num_classes=7):
        super(StyleDiscriminator, self).__init__(num_filter=num_filter)
        self.aag = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x3):
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x_out = self.aag(x8).squeeze()
        class_pred = self.linear_layer(x_out)
        return class_pred


class ClassificationModel(nn.Module):
    """Just an MLP"""
    def __init__(self, input_size=512, num_classes=7):
        super(ClassificationModel, self).__init__()
        self.aag = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        h = self.aag(h).squeeze()
        y = self.layer(h)
        return y
