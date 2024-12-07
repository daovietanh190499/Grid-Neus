import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
import models.gridsample_grad2.cuda_gridsample as grid_sample

class SDFNetwork(nn.Module):
    def __init__(
        self, 
        bound_min, 
        bound_max,
        width, 
        resolution, 
        scale=1.5, 
        sdfnet_width=128, 
        sdfnet_depth=3, 
        rgbnet_width=128, 
        rgbnet_depth=3,
        bias=0.5,
        geometric_init=True
    ):
        super(SDFNetwork, self).__init__()

        if geometric_init:
            linear = nn.Linear(3, width)
            nn.init.constant_(linear.bias, 0.0)
            nn.init.normal_(linear.weight, 0.0, np.sqrt(2) / np.sqrt(width))
            linear.requires_grad_(False)
            for param in linear.parameters():
                param.requires_grad = False

            grid = self.create_coordinate_grid(resolution)
            grid = linear(grid.permute(1, 2, 3, 0).reshape((resolution**3, 3)))
            grid = grid.reshape((resolution, resolution, resolution, width)).permute(3, 0, 1, 2)
            self.register_buffer("grid", grid.unsqueeze(0))  # Store as a non-trainable buffer
        else:
            self.register_buffer("grid", torch.ones((1, width, resolution, resolution, resolution)) * 0.3)

        self.voxel_grid = nn.Parameter(self.grid.clone())
        self.scale = scale
        self.resolution = resolution

        sdfnet_layers = []
        dims = [width + 10*6 + 3, 128] + [sdfnet_width for _ in range(sdfnet_depth-2)] + [1]

        for l in range(len(dims) - 1):
            lin = nn.Linear(dims[l], dims[l+1])
            
            if geometric_init:
                if l == len(dims) - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dims[l+1]))
                    torch.nn.init.constant_(lin.weight[:, -(10*6):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dims[l+1]))

            if l == len(dims) - 2:
                sdfnet_layers.append(lin)
            else:
                sdfnet_layers.append(nn.Sequential(lin, nn.ReLU(inplace=True)))
        
        self.sdfnet = nn.Sequential(*sdfnet_layers)

        self.rgbnet = nn.Sequential(
            nn.Linear(width + 10*6 + 3 + 4*6 + 3 + 3, rgbnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth-2)
            ],
            nn.Linear(rgbnet_width, 3),
        )

    def create_coordinate_grid(self, N):
        coords = torch.linspace(-1, 1, N)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
        grid_coord = torch.stack([x, y, z], dim=0).detach()
        return grid_coord

    def forward(self, x):
        output = torch.zeros((x.shape[0], self.grid.shape[1]), device=x.device)
        mask = (x[:, 0].abs() < self.scale) & (x[:, 1].abs() < self.scale) & (x[:, 2].abs() < self.scale)

        idx = (x[mask]/self.scale).clip(-1, 1)

        tmp = grid_sample.grid_sample_3d(self.voxel_grid, idx.view(1, 1, 1, -1, 3), align_corners=True)
        # tmp = nn.functional.grid_sample(self.voxel_grid, idx.view(1, 1, 1, -1, 3), mode='bilinear', align_corners=True)
        tmp = tmp.permute([0, 2, 3, 4, 1]).squeeze(0).squeeze(0).squeeze(0)
        output[mask] = tmp
        return output

    def sdf(self, x):
        emb_x = self.positional_encoding(x, 10)
        feats = self.forward(x)
        output = self.sdfnet(torch.cat([nn.ReLU()(feats), emb_x], dim = 1))
        sdf = output
        return sdf

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def gradient_sdf_color(self, x, d):
        x_grad = x.clone()
        x_grad.requires_grad_(True)

        emb_x = self.positional_encoding(x, 10)
        emb_x_grad = self.positional_encoding(x_grad, 10)
        emb_d = self.positional_encoding(d, 4)

        feats = self.forward(x)
        feats_grad = self.forward(x_grad)

        sdf = self.sdfnet(torch.cat([nn.ReLU()(feats), emb_x], dim = 1))
        sdf_grad = self.sdfnet(torch.cat([nn.ReLU()(feats_grad), emb_x_grad], dim = 1))

        y = sdf_grad
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x_grad,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        color = torch.sigmoid(self.rgbnet(torch.cat([feats, emb_x, emb_d, gradients.clone().detach()], dim=1)))

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
