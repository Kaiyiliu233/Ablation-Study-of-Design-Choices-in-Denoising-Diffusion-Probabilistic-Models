import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=time.device) / half)
        args = time[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLPEmbeddings(nn.Module):
    # sinusoidal -> linear -> silu -> linear
    def __init__(self, dim):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 1x1 conv for residual when channels change
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.residual_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum("b c i, b c j -> b i j", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("b i j, b c j -> b c i", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, ch_mults=(1, 2, 4),
                 num_res_blocks=2, time_dim=256, attn_at_bottom=True):
        super().__init__()
        self.time_embedding = TimeMLPEmbeddings(time_dim)
        num_levels = len(ch_mults)
        channels = [base_ch * m for m in ch_mults]
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # encoder
        self.downblocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev_ch = channels[0]
        self.skip_channels = []
        for level in range(num_levels):
            ch = channels[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(prev_ch, ch, time_dim))
                prev_ch = ch
                self.skip_channels.append(ch)
            self.downblocks.append(blocks)
            if level < num_levels - 1:
                self.downsamplers.append(Downsample(ch))
            else:
                self.downsamplers.append(nn.Identity())

        # bottleneck
        mid_ch = channels[-1]
        self.mid_blocks1 = ResBlock(mid_ch, mid_ch, time_dim)
        self.mid_attn = SelfAttention(mid_ch) if attn_at_bottom else nn.Identity()
        self.mid_blocks2 = ResBlock(mid_ch, mid_ch, time_dim)

        # decoder
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level in reversed(range(num_levels)):
            ch = channels[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                blocks.append(ResBlock(prev_ch + skip_ch, ch, time_dim))
                prev_ch = ch
            self.up_blocks.append(blocks)
            if level > 0:
                self.upsamplers.append(Upsample(ch))
            else:
                self.upsamplers.append(nn.Identity())

        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_ch, 3, padding=1)

    def forward(self, x, time):
        t_emb = self.time_embedding(time)
        x = self.input_conv(x)

        skips = []
        for blocks, down in zip(self.downblocks, self.downsamplers):
            for blk in blocks:
                x = blk(x, t_emb)
                skips.append(x)
            x = down(x)

        # middle
        x = self.mid_blocks1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_blocks2(x, t_emb)

        for blocks, up in zip(self.up_blocks, self.upsamplers):
            for blk in blocks:
                x = torch.cat([x, skips.pop()], dim=1)
                x = blk(x, t_emb)
            x = up(x)

        return self.out_conv(F.silu(self.out_norm(x)))
