import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

class SDFNetwork(nn.Module):
    def __init__(self, bound_min, bound_max, resolution, scale=1.5, sdfnet_width=128, sdfnet_depth=3, rgbnet_width=128, rgbnet_depth=3):
        super(SDFNetwork, self).__init__()
        self.grid = torch.ones((1, 1 + 128, resolution, resolution, resolution))*0.3
        self.voxel_grid = nn.Parameter(self.grid)
        self.scale = scale
        self.resolution = resolution

        self.sdfnet = nn.Sequential(
            nn.Linear(128 + 10*6 + 3, sdfnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(sdfnet_width, sdfnet_width), nn.ReLU(inplace=True))
                for _ in range(sdfnet_depth-2)
            ],
            nn.Linear(sdfnet_width, 4),
        )

        self.rgbnet = nn.Sequential(
            nn.Linear(128 + 10*6 + 3 + 4*6 + 3 + 3, rgbnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth-2)
            ],
            nn.Linear(rgbnet_width, 3),
        )

    def forward(self, x):
        output = torch.zeros((x.shape[0], self.grid.shape[1]), device=x.device)
        mask = (x[:, 0].abs() < self.scale) & (x[:, 1].abs() < self.scale) & (x[:, 2].abs() < self.scale)

        idx = (x[mask]/self.scale).clip(-1, 1)

        tmp = nn.functional.grid_sample(self.voxel_grid, idx.view(1, 1, 1, -1, 3), mode='bilinear', align_corners=True)
        tmp = tmp.permute([0, 2, 3, 4, 1]).squeeze(0).squeeze(0).squeeze(0)
        output[mask] = tmp
        return output

    def sdf(self, x):
        emb_x = self.positional_encoding(x, 10)
        feats = self.forward(x)
        sdf_color_feat = feats[:, 1:]
        output = self.sdfnet(torch.cat([sdf_color_feat, emb_x], dim = 1))
        sdf = output[:, :1]
        return sdf

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def gradient_sdf_color(self, x, d):
        emb_x = self.positional_encoding(x, 10)
        emb_d = self.positional_encoding(d, 4)
        feats = self.forward(x)
        sdf_inter = feats[:, :1]
        sdf_color_feat = feats[:, 1:]

        output = self.sdfnet(torch.cat([sdf_color_feat, emb_x], dim = 1))
        sdf, gradients = output[:, :1], torch.tanh(output[:, 1:])

        color = torch.sigmoid(self.rgbnet(torch.cat([sdf_color_feat, emb_x, emb_d, gradients], dim=1)))

        return gradients, sdf, color

class NeRF(nn.Module):
    def __init__(self, embedding_dim_pos=20, embedding_dim_direction=8, hidden_dim=128, multires=0, d_in=3):
        super(NeRF, self).__init__()

        self.input_ch = 3
        self.embed_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        self.block1 = nn.Sequential(nn.Linear(self.input_ch, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )

        self.block2 = nn.Sequential(nn.Linear(self.input_ch + hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )

        self.block3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 75), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

        self.bandwidth = nn.Parameter(torch.zeros((1, 25)))
        self.p = nn.Parameter(torch.randn((25, 2)))

    def to_cartesian(self, theta_phi):
        return torch.stack([torch.sin(theta_phi[:, 0]) * torch.cos(theta_phi[:, 1]),
                            torch.sin(theta_phi[:, 0]) * torch.sin(theta_phi[:, 1]),
                            torch.cos(theta_phi[:, 0])], axis=1)

    def forward(self, o, d):
        emb_x = o
        if self.embed_fn is not None:
            emb_x = self.embed_fn(o)

        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(h)
        k = self.block4(h).reshape(o.shape[0], 25, 3)

        c = (k * torch.exp(
            (self.bandwidth.unsqueeze(-1) * self.to_cartesian(self.p).unsqueeze(0) * d.unsqueeze(1)))).sum(1)

        return sigma, c

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
