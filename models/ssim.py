
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=3, data_range=2.0):
        super().__init__()
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = (g[:, None] * g[None, :]).expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer('window', w)
        self.channels = channels
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def forward(self, x, y):
        pad = self.window.shape[-1] // 2
        ch = self.channels
        mu_x = F.conv2d(x, self.window, padding=pad, groups=ch)
        mu_y = F.conv2d(y, self.window, padding=pad, groups=ch)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sig_x = F.conv2d(x * x, self.window, padding=pad, groups=ch) - mu_x2
        sig_y = F.conv2d(y * y, self.window, padding=pad, groups=ch) - mu_y2
        sig_xy = F.conv2d(x * y, self.window, padding=pad, groups=ch) - mu_xy
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)) / \
                   ((mu_x2 + mu_y2 + self.C1) * (sig_x + sig_y + self.C2))
        return 1.0 - ssim_map.mean()
