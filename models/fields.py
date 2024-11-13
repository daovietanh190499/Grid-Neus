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

    def lerp(self, u, a, b):
        return a + u * (b - a)

    def compute_sdf_normal(self, points, s000, s001, s010, s100, s101, s011, s110, s111):
        x, y, z = points[:, :1], points[:, 1:2], points[:, 2:]

        # Partial derivative w.r.t x
        y0 = self.lerp(y, s100 - s000, s110 - s010)
        y1 = self.lerp(y, s101 - s001, s111 - s011)
        df_dx = self.lerp(z, y0, y1)

        # Partial derivative w.r.t y
        x0 = self.lerp(x, s010 - s000, s110 - s100)
        x1 = self.lerp(x, s011 - s001, s111 - s101)
        df_dy = self.lerp(z, x0, x1)

        # Partial derivative w.r.t z
        x0 = self.lerp(x, s001 - s000, s101 - s100)
        x1 = self.lerp(x, s011 - s010, s111 - s110)
        df_dz = self.lerp(y, x0, x1)

        # Normal vector (negative gradient of the SDF)
        normals = torch.hstack((df_dx, df_dy, df_dz))
        
        return normals.detach()

    def get_aabb_cube(self, x):
        abs_x = (x / (2 * self.scale / self.resolution) + self.resolution / 2)
        
        idx_000 = abs_x.long().clip(0, self.resolution - 1)
        idx_001 = idx_000 + torch.tensor([0, 0, 1]).repeat(x.shape[0], 1).to(x.device)
        idx_010 = idx_000 + torch.tensor([0, 1, 0]).repeat(x.shape[0], 1).to(x.device)
        idx_100 = idx_000 + torch.tensor([1, 0, 0]).repeat(x.shape[0], 1).to(x.device)
        idx_101 = idx_000 + torch.tensor([1, 0, 1]).repeat(x.shape[0], 1).to(x.device)
        idx_011 = idx_000 + torch.tensor([0, 1, 1]).repeat(x.shape[0], 1).to(x.device)
        idx_110 = idx_000 + torch.tensor([1, 1, 0]).repeat(x.shape[0], 1).to(x.device)
        idx_111 = idx_000 + torch.tensor([1, 1, 1]).repeat(x.shape[0], 1).to(x.device)

        rel_x = abs_x - idx_000

        return abs_x, rel_x, idx_000, idx_001, idx_010, idx_100, idx_101, idx_011, idx_110, idx_111

    def get_aabb_sdf(self, idx_000, idx_001, idx_010, idx_100, idx_101, idx_011, idx_110, idx_111):
        s_000 = self.voxel_grid[0, :1, idx_000[:, 0], idx_000[:, 1], idx_000[:, 2]].permute([1, 0])
        s_001 = self.voxel_grid[0, :1, idx_001[:, 0], idx_001[:, 1], idx_001[:, 2]].permute([1, 0])
        s_010 = self.voxel_grid[0, :1, idx_010[:, 0], idx_010[:, 1], idx_010[:, 2]].permute([1, 0])
        s_100 = self.voxel_grid[0, :1, idx_100[:, 0], idx_100[:, 1], idx_100[:, 2]].permute([1, 0])
        s_101 = self.voxel_grid[0, :1, idx_101[:, 0], idx_101[:, 1], idx_101[:, 2]].permute([1, 0])
        s_011 = self.voxel_grid[0, :1, idx_011[:, 0], idx_011[:, 1], idx_011[:, 2]].permute([1, 0])
        s_110 = self.voxel_grid[0, :1, idx_110[:, 0], idx_110[:, 1], idx_110[:, 2]].permute([1, 0])
        s_111 = self.voxel_grid[0, :1, idx_111[:, 0], idx_111[:, 1], idx_111[:, 2]].permute([1, 0])

        return s_000, s_001, s_010, s_100, s_101, s_011, s_110, s_111
        return x[~mask, :], d[~mask, :], s000[~mask], s001[~mask], s010[~mask], s100[~mask], s101[~mask], s011[~mask], s110[~mask], s111[~mask]

    def get_polynomial_coeff(self, o, d, s_000, s_001, s_010, s_100, s_101, s_011, s_110, s_111):
        ox, oy, oz = o[:, :1], o[:, 1:2], o[:, 2:]
        dx, dy, dz = d[:, :1], d[:, 1:2], d[:, 2:]
        a = s_101 - s_001
        k0 = s_000
        k1 = s_100 - s_000
        k2 = s_010 - s_000
        k3 = s_110 - s_010 - k1
        k4 = k0 - s_001
        k5 = k1 - a
        k6 = k2 - (s_011 - s_001)
        k7 = k3 - (s_111 - s_011 - a)

        m0 = ox * oy
        m1 = dx * dy
        m2 = ox * dy + oy * dx
        m3 = k5 * oz - k1
        m4 = k6 * oz - k2
        m5 = k7 * oz - k3

        c0 = (k4 * oz - k0) + ox * m3 + oy * m4 + m0 * m5
        c1 = (dx * m3 + dy * m4 + m2 * m5 + dz * (k4 + k5 * ox + k6 * oy + k7 * m0))
        c2 = m1 * m5 + dz * (k5 * dx + k6 * dy + k7 * m2)
        c3 = k7 * m1 * dz

        # return c3.squeeze(1), c2.squeeze(1), c1.squeeze(1), c0.squeeze(1)
        return c3.squeeze(1).detach(), c2.squeeze(1).detach(), c1.squeeze(1).detach(), c0.squeeze(1).detach()

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

        # abs_x, rel_x, *idxs = self.get_aabb_cube(x)
        # s_idx = self.get_aabb_sdf(*idxs)
        # gradients = self.compute_sdf_normal(rel_x, *s_idx)

        output = self.sdfnet(torch.cat([sdf_color_feat, emb_x], dim = 1))
        sdf, gradients = output[:, :1], torch.tanh(output[:, 1:])

        # x.requires_grad_(True)
        # y = self.sdf(x)
        # d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # gradients = torch.autograd.grad(
        #     outputs=y,
        #     inputs=x,
        #     grad_outputs=d_output,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]

        color = torch.sigmoid(self.rgbnet(torch.cat([sdf_color_feat, emb_x, emb_d, gradients], dim=1)))

        return gradients, sdf, color


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
