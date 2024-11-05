import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import svox

# from torch.utils.cpp_extension import load
# grid_sample_cuda = load(
#     name="grid_sample_cuda",
#     sources=["grid_sample.cpp", "grid_sample_cuda.cu"],
#     verbose=True
# )


def eval_spherical_function(k, d):
    x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]

    # Modified from https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
    return 0.282095 * k[..., 0] + \
        - 0.488603 * y * k[..., 1] + 0.488603 * z * k[..., 2] - 0.488603 * x * k[..., 3] + \
        (1.092548 * x * y * k[..., 4] - 1.092548 * y * z * k[..., 5] + 0.315392 * (2.0 * z * z - x * x - y * y) * k[
               ..., 6] + -1.092548 * x * z * k[..., 7] + 0.546274 * (x * x - y * y) * k[..., 8])


# class NerfModel(nn.Module):
#     def __init__(self, N=256, scale=1.5):
#         """
#         :param N
#         :param scale: The maximum absolute value among all coordinates for objects in the scene
#         """
#         super(NerfModel, self).__init__()

#         self.voxel_grid = nn.Parameter(torch.ones((1, 27+ 1, N, N, N)) / 100)
#         self.scale = scale
#         self.N = N

#     def forward(self, x, d):
#         color = torch.zeros_like(x)
#         sigma = torch.zeros((x.shape[0]), device=x.device)
#         mask = (x[:, 0].abs() < self.scale) & (x[:, 1].abs() < self.scale) & (x[:, 2].abs() < self.scale)

#         # idx = (x[mask] / (2 * self.scale / self.N) + self.N / 2).long().clip(0, self.N - 1)
#         idx = (x[mask]/self.scale).clip(-1, 1)
#         # tmp = self.voxel_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
#         tmp = nn.functional.grid_sample(self.voxel_grid, idx.view(1, 1, 1, -1, 3), mode='bilinear', align_corners=True)
#         tmp = tmp.permute([0, 2, 3, 4, 1]).squeeze(0).squeeze(0).squeeze(0)
#         sigma[mask], k = torch.nn.functional.relu(tmp[:, 0]), tmp[:, 1:]
#         color[mask] = eval_spherical_function(k.reshape(-1, 3, 9), d[mask])
#         return color, sigma

class NerfModel(nn.Module):
    def __init__(self, resolution=256, scale=1.5):
        super(NerfModel, self).__init__()

        self.voxel_grid = nn.Parameter(torch.ones((1, 27 + 1, resolution, resolution, resolution)) / 100)
        self.scale = scale
        self.resolution = resolution
        self.activation = nn.ReLU()

    def forward(self, x, d):
        output = torch.zeros((x.shape[0], 28), device=x.device)
        mask = (x[:, 0].abs() < self.scale) & (x[:, 1].abs() < self.scale) & (x[:, 2].abs() < self.scale)

        idx = (x[mask]/self.scale).clip(-1, 1)

        # tmp = grid_sample_cuda(self.voxel_grid[0].permute(1, 2, 3, 0), idx.view(-1, 3))

        tmp = nn.functional.grid_sample(self.voxel_grid, idx.view(1, 1, 1, -1, 3), mode='bilinear', align_corners=True)
        tmp = tmp.permute([0, 2, 3, 4, 1]).squeeze(0).squeeze(0).squeeze(0)
        output[mask] = tmp
        sigma = self.activation(output[:, :1])
        c = self.eval_spherical_function(output[:, 1:].reshape(-1, 3, 9), d)
        return sigma, c

    def sdf(self, x):
        return self.activation(self.forward(x)[:, :1])

    def color(self, x, d):
        c = self.eval_spherical_function(self.forward(x)[:, 1:].reshape(-1, 3, 9), d)
        return c

    def eval_spherical_function(self, k, d):
        x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]

        # Modified from https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
        return 0.282095 * k[..., 0] + \
            - 0.488603 * y * k[..., 1] + 0.488603 * z * k[..., 2] - 0.488603 * x * k[..., 3] + \
            (1.092548 * x * y * k[..., 4] - 1.092548 * y * z * k[..., 5] + 0.315392 * (2.0 * z * z - x * x - y * y) * k[
                ..., 6] + -1.092548 * x * z * k[..., 7] + 0.546274 * (x * x - y * y) * k[..., 8])

    def lerp(self, u, a, b):
        return a + u * (b - a)

    def compute_sdf_normal(self, points, sdf_values):
        x, y, z = points[:, :1], points[:, 1:2], points[:, 2:]
        # Unpack SDF values
        s000, s001, s010, s100, s101, s011, s110, s111 = sdf_values

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
        
        return normals

    def gradient_old(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def gradient(self, x):
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
        s_000 = self.voxel_grid[0, :1, idx_000[:, 0], idx_000[:, 1], idx_000[:, 2]].permute([1, 0])
        s_001 = self.voxel_grid[0, :1, idx_001[:, 0], idx_001[:, 1], idx_001[:, 2]].permute([1, 0])
        s_010 = self.voxel_grid[0, :1, idx_010[:, 0], idx_010[:, 1], idx_010[:, 2]].permute([1, 0])
        s_100 = self.voxel_grid[0, :1, idx_100[:, 0], idx_100[:, 1], idx_100[:, 2]].permute([1, 0])
        s_101 = self.voxel_grid[0, :1, idx_101[:, 0], idx_101[:, 1], idx_101[:, 2]].permute([1, 0])
        s_011 = self.voxel_grid[0, :1, idx_011[:, 0], idx_011[:, 1], idx_011[:, 2]].permute([1, 0])
        s_110 = self.voxel_grid[0, :1, idx_110[:, 0], idx_110[:, 1], idx_110[:, 2]].permute([1, 0])
        s_111 = self.voxel_grid[0, :1, idx_111[:, 0], idx_111[:, 1], idx_111[:, 2]].permute([1, 0])

        return self.compute_sdf_normal(rel_x, (s_000, s_001, s_010, s_100, s_101, s_011, s_110, s_111))



@torch.no_grad()
def test(model, hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'Imgs/img_{img_index}.png', bbox_inches='tight')
    plt.close()


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    sigma, colors = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192):
    total_loss = []
    for _ in range(nb_epochs):
        training_loss = []
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = torch.nn.functional.mse_loss(ground_truth_px_values, regenerated_px_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            if(len(training_loss) % 170 == 0):
                if len(total_loss) > 0:
                    print(training_loss)
                total_loss.append(training_loss)
                training_loss = []

        scheduler.step()
        print(training_loss)
    return training_loss


if __name__ == "__main__":
    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('/home/coder/psrnet/nerf_datasets/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('/home/coder/psrnet/nerf_datasets/testing_data.pkl', allow_pickle=True))

    model = NerfModel(resolution=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    data_loader = DataLoader(training_dataset, batch_size=2048, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=1, device=device, hn=2, hf=6, nb_bins=192)
    for img_index in [0, 60, 120, 180]:
        test(model, 2, 6, testing_dataset, img_index=img_index, nb_bins=192, H=400, W=400)

# https://arxiv.org/pdf/2303.14158