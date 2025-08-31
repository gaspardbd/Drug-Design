from __future__ import annotations

from typing import Iterable, Tuple, Union

import math
import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Create sinusoidal timestep embeddings as in transformer/denoising models.

    Produces embeddings of size embedding_dim for integer timesteps.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:  # [B]
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, device=device).float() / (half_dim - 1)
        emb = torch.exp(exponent)  # [half]
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb  # [B, D]


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: Union[int, None]) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.time_mlp: nn.Module | None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch),
            )
        else:
            self.time_mlp = None
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        if self.time_mlp is not None and t_emb is not None:
            # Add time embedding as bias
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = self.k(x_norm).reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v(x_norm).reshape(b, self.num_heads, c // self.num_heads, h * w)
        attn = torch.softmax(torch.einsum("b h c n, b h c m -> b h n m", q, k) / math.sqrt(c // self.num_heads), dim=-1)
        out = torch.einsum("b h n m, b h c m -> b h c n", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Unet(nn.Module):
    """Small UNet for 28x28 Fashion-MNIST with optional time embeddings.

    The signature is made compatible with the notebook's instantiation.
    """

    def __init__(
        self,
        dim: int = 28,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        channels: int = 1,
        with_time_emb: bool = True,
        convnext_mult: int = 2,
    ) -> None:
        super().__init__()
        base_channels = 32 if init_dim is None else init_dim
        time_emb_dim = base_channels * 4 if with_time_emb else None
        self.time_embedding: nn.Module | None
        if with_time_emb:
            self.time_embedding = nn.Sequential(
                SinusoidalPositionEmbeddings(base_channels),
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_embedding = None

        self.init_conv = conv3x3(channels, base_channels)

        # Down path
        down_channels = []
        cur_ch = base_channels
        downs = []
        for mult in dim_mults:
            out_ch = base_channels * mult
            downs.append(ResidualBlock(cur_ch, out_ch, time_emb_dim))
            downs.append(ResidualBlock(out_ch, out_ch, time_emb_dim))
            downs.append(AttentionBlock(out_ch))
            down_channels.append(out_ch)
            cur_ch = out_ch
            if mult != dim_mults[-1]:
                downs.append(Downsample(cur_ch))
        self.downs = nn.ModuleList(downs)

        # Middle
        mid_ch = cur_ch
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim)

        # Up path
        ups = []
        for mult in reversed(dim_mults):
            out_ch = base_channels * mult
            ups.append(ResidualBlock(cur_ch + out_ch, out_ch, time_emb_dim))
            ups.append(ResidualBlock(out_ch, out_ch, time_emb_dim))
            ups.append(AttentionBlock(out_ch))
            cur_ch = out_ch
            if mult != dim_mults[0]:
                ups.append(Upsample(cur_ch))
        self.ups = nn.ModuleList(ups)

        default_out_dim = channels
        self.final_norm = nn.GroupNorm(8, cur_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(cur_ch, default_out_dim if out_dim is None else out_dim, 1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps) if self.time_embedding is not None else None
        x = self.init_conv(x)

        skips = []
        i = 0
        while i < len(self.downs):
            block1: ResidualBlock = self.downs[i]  # type: ignore[assignment]
            block2: ResidualBlock = self.downs[i + 1]  # type: ignore[assignment]
            attn: AttentionBlock = self.downs[i + 2]  # type: ignore[assignment]
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            skips.append(x)
            i += 3
            if i < len(self.downs) and isinstance(self.downs[i], Downsample):
                x = self.downs[i](x)  # type: ignore[index]
                i += 1

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        j = 0
        while j < len(self.ups):
            block1: ResidualBlock = self.ups[j]  # type: ignore[assignment]
            block2: ResidualBlock = self.ups[j + 1]  # type: ignore[assignment]
            attn: AttentionBlock = self.ups[j + 2]  # type: ignore[assignment]
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            j += 3
            if j < len(self.ups) and isinstance(self.ups[j], Upsample):
                x = self.ups[j](x)  # type: ignore[index]
                j += 1

        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x


