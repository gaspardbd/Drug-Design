import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

from ddpm.data import get_fashion_mnist_dataloader, save_image_grid
from ddpm.diffusion import q_sample, sampling
from ddpm.model import Unet
from ddpm.schedules import (
    get_diffusion_constants,
    linear_beta_schedule,
    cosine_beta_schedule,
)
from ddpm.utils import ensure_dir, save_sampling_gif


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDPM on Fashion-MNIST")
    p.add_argument("--data_path", type=str, default="fashion_mnist", help="HF datasets path or 'fashion_mnist'")
    p.add_argument("--out_dir", type=str, default="./outputs/fashion_ddpm", help="Output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, choices=["linear", "cosine"], default="linear")
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    ckpt_dir = str(Path(args.out_dir) / "checkpoints")
    img_dir = str(Path(args.out_dir) / "images")
    ensure_dir(ckpt_dir)
    ensure_dir(img_dir)

    dataloader, channels, image_size, _ = get_fashion_mnist_dataloader(args.data_path, args.batch_size)

    model = Unet(
        dim=image_size,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=channels,
        with_time_emb=True,
        convnext_mult=2,
    ).to(args.device)

    criterion = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    schedule_fn = linear_beta_schedule if args.schedule == "linear" else cosine_beta_schedule
    constants = get_diffusion_constants(args.timesteps, schedule_fn)
    # Keep constants on CPU; they are small and moved per-step

    for epoch in range(1, args.epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        model.train()
        for batch in loop:
            optimizer.zero_grad(set_to_none=True)
            batch_images = batch["pixel_values"].to(args.device)
            bsz = batch_images.shape[0]
            batch_t = torch.randint(0, args.timesteps, (bsz,), device=args.device).long()
            noise = torch.randn_like(batch_images)
            x_noisy = q_sample(constants, batch_images, batch_t, noise=noise)
            predicted_noise = model(x_noisy, batch_t)
            loss = criterion(noise, predicted_noise)
            loop.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

        if epoch % args.save_every == 0:
            ckpt_path = str(Path(ckpt_dir) / f"epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, ckpt_path)

        # Quick sample at end of epoch
        model.eval()
        with torch.no_grad():
            frames = sampling(
                model,
                (args.batch_size, channels, image_size, image_size),
                args.timesteps,
                constants,
                args.device,
            )
        final = frames[-1]
        img_path = str(Path(img_dir) / f"samples_epoch_{epoch}.png")
        save_image_grid(final, img_path)
        gif_path = str(Path(img_dir) / f"sampling_epoch_{epoch}.gif")
        save_sampling_gif(frames, gif_path)


if __name__ == "__main__":
    main()


