from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.utils import make_grid

from .data import normalize_im


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_sampling_gif(frames: List[torch.Tensor], out_path: str, step: int = 5, duration: int = 10) -> None:
    to_pil_frames = [
        _tensor_grid_to_pil(tensor_frame) for tensor_frame in frames[:: max(1, step)]
    ]
    if not to_pil_frames:
        return
    first = to_pil_frames[0]
    first.save(out_path, format="GIF", append_images=to_pil_frames[1:], save_all=True, duration=duration, loop=0)


def _tensor_grid_to_pil(tens_im: torch.Tensor) -> Image.Image:
    grid = make_grid(normalize_im(tens_im))
    ndarr = (grid.clamp(0, 1).mul(255).permute(1, 2, 0).byte().cpu().numpy())
    return Image.fromarray(ndarr)


