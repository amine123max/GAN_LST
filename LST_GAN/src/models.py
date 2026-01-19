import torch
import torch.nn as nn

class Generator(nn.Module):
    """Conditional Generator: Takes 10 previous years LST as input, generates next year LST"""
    def __init__(self, nc, ngf, num_input_years=10):
        super(Generator, self).__init__()
        # Encoder (downsample)
        # Input channels = nc * num_input_years (10 years of temperature data)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(nc * num_input_years, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder (upsample)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net style architecture with skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        d1 = self.decoder1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        output = self.decoder4(d3)
        
        return output

class Discriminator(nn.Module):
    """Conditional Discriminator: Takes concatenated (10 previous years, current year) as input"""
    def __init__(self, nc, ndf, num_input_years=10):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc*(num_input_years+1)) x 64 x 64
            nn.Conv2d(nc * (num_input_years + 1), ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # Concatenate condition (x) and target (y)
        input_pair = torch.cat([x, y], dim=1)
        return self.main(input_pair)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
