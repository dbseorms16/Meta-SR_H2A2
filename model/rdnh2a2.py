# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
import numpy as np
import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDNH2A2SR(args)

class H2A2SR(nn.Module):
    def __init__(self, opt, num_channels=3, conv=common.default_conv):
        super(H2A2SR, self).__init__()
        
        self.scale = opt.int_scale + opt.float_scale
        self.res_scale = opt.scale[0] / opt.int_scale

        self.opt = opt
        self.n_feats = opt.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        self.RCAB1 = common.RCAB(conv,32, kernel_size, act=act)
        self.RCAB2 = common.RCAB(conv, 64, kernel_size, act=act)
        self.RCAB3 = common.RCAB(conv, 128, kernel_size, act=act)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(256, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.tail = nn.Conv2d(num_channels , num_channels, kernel_size=3, padding= 3 // 2)
    
    def dct_2d(self, x, norm=None):
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm=None):
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

    def _rfft(self, x, signal_ndim=1, onesided=True):
        # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
        # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
        # torch 1.8.0 torch.fft.rfft to torch 1.5.0 torch.rfft as signal_ndim=1
        # written by mzero
        odd_shape1 = (x.shape[1] % 2 != 0)
        x = torch.fft.rfft(x)
        x = torch.cat([x.real.unsqueeze(dim=2), x.imag.unsqueeze(dim=2)], dim=2)
        if onesided == False:
            _x = x[:, 1:, :].flip(dims=[1]).clone() if odd_shape1 else x[:, 1:-1, :].flip(dims=[1]).clone()
            _x[:,:,1] = -1 * _x[:,:,1]
            x = torch.cat([x, _x], dim=1)
        return x

    def _irfft(self, x, signal_ndim=1, onesided=True):
        # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
        # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
        # torch 1.8.0 torch.fft.irfft to torch 1.5.0 torch.irfft as signal_ndim=1
        # written by mzero
        if onesided == False:
            res_shape1 = x.shape[1]
            x = x[:,:(x.shape[1] // 2 + 1),:]
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x, n=res_shape1)
        else:
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x)
        return x

    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = self._rfft(v, 1, onesided=False)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def idct(self, X, norm=None):
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = self._irfft(V, 1, onesided=False)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)

    def forward(self, x):

        N, C, H, W = x.size()
        outH, outW = int(H*self.res_scale), int(W*self.res_scale)
        x = self.dct_2d(x)
        zeroPad2d = nn.ZeroPad2d((0,int(outW - W), 0, int(outH - H))).to('cuda:0')
        x = zeroPad2d(x)
        x = self.idct_2d(x)
        x = self.conv1(x)
        copyx = x
        x = self.RCAB1(x)
        x = torch.cat((x, copyx), 1)
        x = self.RCAB2(x)
        copyx2 = x
        x = torch.cat((x, copyx2), 1)
        x = self.RCAB3(x)
        copyx3 = x
        x = torch.cat((x, copyx3), 1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tail(x)
        return x


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDNH2A2SR(nn.Module):
    def __init__(self, args):
        super(RDNH2A2SR, self).__init__()
        r = 2
        G0 = args.G0
        kSize = args.RDNkSize

        self.H2A2SR = H2A2SR(args)
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]


        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        x = self.UPNet(x)
        x = self.add_mean(x)
        x = self.H2A2SR(x)
        return x
