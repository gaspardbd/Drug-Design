from typing import List, Tuple, Union

import torch
from torch import nn
from tqdm.auto import tqdm


def extract_by_timestep(constant_vec: torch.Tensor, batch_t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather per-example constants for given timesteps and reshape for broadcasting.

    Args:
        constant_vec: 1D tensor of shape [T]
        batch_t: int64 tensor of shape [B]
        x_shape: target tensor shape, e.g., [B, C, H, W]
    Returns:
        Tensor of shape [B, 1, 1, 1] (or matching dims for x_shape) on same device as batch_t.
    """
    b = batch_t.shape[0]
    out = constant_vec.gather(0, batch_t.cpu()).to(batch_t.device)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def q_sample(constants: dict, batch_x0: torch.Tensor, batch_t: torch.Tensor, noise: Union[torch.Tensor, None] = None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(batch_x0)
    sqrt_alphas_cumprod_t = extract_by_timestep(constants["sqrt_alphas_cumprod"], batch_t, batch_x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_by_timestep(
        constants["sqrt_one_minus_alphas_cumprod"], batch_t, batch_x0.shape
    )
    return sqrt_alphas_cumprod_t * batch_x0 + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(constants: dict, batch_xt: torch.Tensor, predicted_noise: torch.Tensor, batch_t: torch.Tensor) -> torch.Tensor:
    betas_t = extract_by_timestep(constants["betas"], batch_t, batch_xt.shape).to(batch_xt.device)
    sqrt_one_minus_alphas_cumprod_t = extract_by_timestep(
        constants["sqrt_one_minus_alphas_cumprod"], batch_t, batch_xt.shape
    ).to(batch_xt.device)
    sqrt_recip_alphas_t = extract_by_timestep(constants["sqrt_recip_alphas"], batch_t, batch_xt.shape).to(
        batch_xt.device
    )

    model_mean = sqrt_recip_alphas_t * (batch_xt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    predicted_image = torch.zeros_like(batch_xt)
    t_zero_index = batch_t == 0

    posterior_variance_t = extract_by_timestep(constants["posterior_variance"], batch_t, batch_xt.shape)
    noise = torch.randn_like(batch_xt)
    predicted_image[~t_zero_index] = model_mean[~t_zero_index] + (
        torch.sqrt(posterior_variance_t[~t_zero_index]) * noise[~t_zero_index]
    )
    predicted_image[t_zero_index] = model_mean[t_zero_index]
    return predicted_image


@torch.no_grad()
def sampling(model: nn.Module, shape: Tuple[int, int, int, int], T: int, constants: dict, device: str) -> List[torch.Tensor]:
    b = shape[0]
    batch_xt = torch.randn(shape, device=device)
    batch_t = torch.full((b,), T, dtype=torch.int64, device=device)

    imgs: List[torch.Tensor] = []
    for _t in tqdm(reversed(range(0, T)), desc="sampling loop time step", total=T):
        batch_t = batch_t - 1
        predicted_noise = model(batch_xt, batch_t)
        batch_xt = p_sample(constants, batch_xt, predicted_noise, batch_t)
        imgs.append(batch_xt.detach().cpu())
    return imgs


