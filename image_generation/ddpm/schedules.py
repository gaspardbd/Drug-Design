import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Linear beta schedule from DDPM.

    Args:
        timesteps: Number of diffusion steps T.
        beta_start: Initial beta value.
        beta_end: Final beta value.
    Returns:
        1D tensor of shape [T] with betas in [0, 1).
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from improved DDPM.

    Args:
        timesteps: Number of diffusion steps T.
        s: Small offset to prevent singularities.
    Returns:
        1D tensor of shape [T] with betas clipped to [1e-4, 0.9999].
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


def get_diffusion_constants(timesteps: int, schedule_fn=cosine_beta_schedule) -> dict:
    """Compute constants used across diffusion steps.

    Returns a dict with:
        betas, sqrt_recip_alphas, sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod, posterior_variance
    """
    betas = schedule_fn(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


