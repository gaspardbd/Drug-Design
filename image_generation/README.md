### Fashion-MNIST DDPM (Python scripts)

Run training and sampling from the command line.

#### Install dependencies
```bash
pip install -r requirements.txt
```

If you prefer, install manually: `torch`, `torchvision`, `datasets`, `tqdm`, `Pillow`, `numpy`.

#### Train and sample
From the `image_generation` directory:
```bash
cd image_generation
python train_fashion_mnist_ddpm.py \
  --data_path fashion_mnist \
  --out_dir ./outputs/fashion_ddpm \
  --epochs 3 \
  --batch_size 64 \
  --timesteps 1000 \
  --schedule linear
```

Notes:
- To use a pre-downloaded HF dataset directory, pass its path via `--data_path` (e.g. `/gpfsdswork/dataset/HuggingFace/fashion_mnist/fashion_mnist/`).
- Outputs are saved under `--out_dir`:
  - `checkpoints/epoch_*.pt` (model checkpoints)
  - `images/samples_epoch_*.png` (image grids)
  - `images/sampling_epoch_*.gif` (sampling process)


