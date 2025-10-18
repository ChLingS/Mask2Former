import torch
import torch.nn as nn


# 改进的U-Net实现（带残差连接）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, 64),
            ResidualBlock(64, 64)
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256)
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512)
        )
        self.enc5 = nn.Sequential(
            ResidualBlock(512, 1024),
            ResidualBlock(1024, 1024)
        )
        
        self.pool = nn.MaxPool2d(2)
        
        # 解码器
        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = nn.Sequential(
            ResidualBlock(1024, 512),
            ResidualBlock(512, 512)
        )
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(
            ResidualBlock(512, 256),
            ResidualBlock(256, 256)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            ResidualBlock(256, 128),
            ResidualBlock(128, 128)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            ResidualBlock(128, 64),
            ResidualBlock(64, 64)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )
        
    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # 解码路径
        d5 = self.up5(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out
