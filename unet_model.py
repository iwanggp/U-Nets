#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File:unet_model.py  
@IDE    ：PyCharm
@Author ：gpwang
@Date   ：2020/9/11
@Desc   ：
=================================================='''
from unet.unet_module import *


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.features = 16  # 定义特征基数
        self.n_channels = 3
        self.n_classes = 1
        self.inc = DoubleConv(self.n_channels, self.features)
        self.down1 = Down(self.features, self.features * 2)
        self.down2 = Down(self.features * 2, self.features * 4)
        self.down3 = Down(self.features * 4, self.features * 8)
        self.down4 = Down(self.features * 8, self.features * 16)

        self.up1 = Up(self.features * 16, self.features * 8)
        self.up2 = Up(self.features * 8, self.features * 4)
        self.up3 = Up(self.features * 4, self.features * 2)
        self.up4 = Up(self.features * 2, self.features)
        self.outc = OutConv(self.features, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    inc = torch.randn(1, 3, 512, 512)
    net = UNet()
    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)
    out = net(inc)
    print(out.size())
    # summary(net, input_size=(3, 512, 512))
