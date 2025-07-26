# model.py
import torch.nn as nn
import torch

class UNetDown(nn.Module):
  def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
    super(UNetDown, self).__init__()

    layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)]
    if normalize:
      layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))

    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)


class UNetUp(nn.Module):
  def __init__(self, in_channels, out_channels, dropout=0.0):
    super(UNetUp, self).__init__()

    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)]
    layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.ReLU(inplace = True))

    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)

  def forward(self, x, skip_input):
    x = self.model(x)
    x = torch.cat((x, skip_input), 1)

    return x
  
class GeneratorUNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3):
    super(GeneratorUNet, self).__init__()

    self.down1 = UNetDown(in_channels, 64, normalize=False) # 출력: [64 X 128 X 128]
    self.down2 = UNetDown(64, 128) # 출력: [128 X 64 X 64]
    self.down3 = UNetDown(128, 256) # 출력: [256 X 32 X 32]
    self.down4 = UNetDown(256, 512, dropout=0.5) # 출력: [512 X 16 X 16]
    self.down5 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 8 X 8]
    self.down6 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 4 X 4]
    self.down7 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 2 X 2]
    self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) # 출력: [512 X 1 X 1]


    # Skip Connection 사용(출력 채널의 크기 X 2 == 다음 입력 채널의 크기)
    self.up1 = UNetUp(512, 512, dropout=0.5) # 출력: [1024 X 2 X 2]
    self.up2 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 4 X 4]
    self.up3 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 8 X 8]
    self.up4 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 16 X 16]
    self.up5 = UNetUp(1024, 256) # 출력: [512 X 32 X 32]
    self.up6 = UNetUp(512, 128) # 출력: [256 X 64 X 64]
    self.up7 = UNetUp(256, 64) # 출력: [128 X 128 X 128]


    self.final = nn.Sequential(
        nn.Upsample(scale_factor = 2),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(128, out_channels, kernel_size = 4, padding = 1),
        nn.Tanh()
    )

  def forward(self, x):
    # 인코더부터 디코더까지 순전파하는 U-Net 생성자(Generator)
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    d6 = self.down6(d5)
    d7 = self.down7(d6)
    d8 = self.down8(d7)
    u1 = self.up1(d8, d7)
    u2 = self.up2(u1, d6)
    u3 = self.up3(u2, d5)
    u4 = self.up4(u3, d4)
    u5 = self.up5(u4, d3)
    u6 = self.up6(u5, d2)
    u7 = self.up7(u6, d1)

    return self.final(u7)

  def __init__(self, in_channels=3, out_channels=3):
    super(GeneratorUNet, self).__init__()
    self.down1 = UNetDown(in_channels, 64, normalize=False)
    self.down2 = UNetDown(64, 128)
    self.down3 = UNetDown(128, 256)
    self.down4 = UNetDown(256, 512, dropout=0.5)
    self.down5 = UNetDown(512, 512, dropout=0.5)
    self.down6 = UNetDown(512, 512, dropout=0.5)
    self.down7 = UNetDown(512, 512, dropout=0.5)
    self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
    self.up1 = UNetUp(512, 512, dropout=0.5)
    self.up2 = UNetUp(1024, 512, dropout=0.5)
    self.up3 = UNetUp(1024, 512, dropout=0.5)
    self.up4 = UNetUp(1024, 512, dropout=0.5)
    self.up5 = UNetUp(1024, 256)
    self.up6 = UNetUp(512, 128)
    self.up7 = UNetUp(256, 64)
    self.final = nn.Sequential(
        nn.Upsample(scale_factor = 2),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(128, out_channels, kernel_size = 4, padding = 1),
        nn.Tanh()
    )
  def forward(self, x):
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    d6 = self.down6(d5)
    d7 = self.down7(d6)
    d8 = self.down8(d7)
    u1 = self.up1(d8, d7)
    u2 = self.up2(u1, d6)
    u3 = self.up3(u2, d5)
    u4 = self.up4(u3, d4)
    u5 = self.up5(u4, d3)
    u6 = self.up6(u5, d2)
    u7 = self.up7(u6, d1)
    return self.final(u7)

class Discriminator(nn.Module):
  def __init__(self, in_channels = 3):
    super(Discriminator, self).__init__()

    def discriminator_block(in_channels, out_channels, normalization = True):
      layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)]
      if normalization:
        layers.append(nn.InstanceNorm2d(out_channels))
      layers.append(nn.LeakyReLU(0.2, inplace = True))
      return layers

    self.model = nn.Sequential(
      *discriminator_block(in_channels * 2, 64, normalization = False),
      *discriminator_block(64, 128),
      *discriminator_block(128, 256), # 출력: [256 X 32 X 32]
      *discriminator_block(256, 512), # 출력: [512 X 16 X 16]
      nn.ZeroPad2d((1, 0, 1, 0)),
      nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False) # 출력: [1 X 16 X 16]
    )

  def forward(self, img_A, img_B):
    img_input = torch.cat((img_A, img_B), 1)
    return self.model(img_input)
