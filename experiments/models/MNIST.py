import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_hidden, n_embedd):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, n_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(n_hidden, n_hidden * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(n_hidden * 2, n_hidden * 4, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(n_hidden * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )
        self.fc = nn.Linear(1024, n_embedd)
        self.n_embedd = n_embedd

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        z = self.net(x)
        z = self.fc(z.view(-1, 1024))
        return z


class Decoder(nn.Module):
    def __init__(self, n_hidden, n_embedd):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_embedd, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(1024, n_hidden * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(n_hidden * 8, n_hidden * 4, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(n_hidden * 4, n_hidden * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(n_hidden * 2, 1, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            # nn.Relu()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        h = F.relu(self.fc1(z))
        deconv_input = self.fc2(h)
        deconv_input = deconv_input.view(-1, 1024, 1, 1)
        # print("deconv_input", deconv_input.size())
        return self.net(deconv_input)


class Encoder2(nn.Module):
    def __init__(self, n_hidden, n_embedd):
        super(Encoder2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, n_embedd),
        )

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        z = self.net(x)
        return z


class Decoder2(nn.Module):
    def __init__(self, n_hidden, n_embedd):
        super(Decoder2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedd, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
