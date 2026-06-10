# AIModel

Experimental real-time generative world project.

## Current milestones

### Tree GAN v0

A compact 128×128 DCGAN that generates isolated trees on black backgrounds.

Current measured performance on an RTX 4070 Super:

- raw generator throughput: about 700 images/s
- live Tkinter preview: about 90–100 FPS
- 65 cleaned RGB training images
- 16 validation images

The loader explicitly excludes filenames ending in `_mask.png`.

### Single-tree orbit v0

An angle-conditioned neural renderer trained on 360 renders of one tree.

Dataset layout:

```text
data/orbit_tree_001/
├── 0001.png   # 0 degrees
├── 0002.png   # 1 degree
├── ...
└── 0360.png   # 359 degrees
```

The model input is the cyclic angle code:

```text
sin(angle), cos(angle)
```

and the output is the corresponding 128×128 view of the tree.

## Setup

Use Python 3.11 or 3.12.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build, then:

```powershell
pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Train the random tree generator

```powershell
python train.py --data "data\train" --epochs 1000 --batch-size 16
```

Samples are written to `runs/tree_gan/samples/` and the latest checkpoint to `runs/tree_gan/checkpoints/latest.pt`.

## Generate random trees

```powershell
python generate.py --checkpoint "runs/tree_gan/checkpoints/latest.pt" --out generated_trees.png
```

## Benchmark random-tree inference

```powershell
python benchmark_fps.py --checkpoint "runs/tree_gan/checkpoints/latest.pt"
```

## Live random-tree preview

```powershell
python live_preview.py --checkpoint "runs/tree_gan/checkpoints/latest.pt"
```

## Train the 360-degree orbit model

Place the 360 renders in `data/orbit_tree_001/`, named exactly `0001.png` through `0360.png`.

Then run:

```powershell
python train_orbit.py --data "data\orbit_tree_001" --epochs 2000 --batch-size 32
```

Orbit samples are written to:

```text
runs/orbit_tree/samples/
```

The latest orbit checkpoint is written to:

```text
runs/orbit_tree/checkpoints/latest.pt
```

## Interactive orbit preview

```powershell
python live_orbit.py --checkpoint "runs/orbit_tree/checkpoints/latest.pt"
```

Controls:

- Left / Right arrows: rotate one degree
- Space: play or pause automatic orbit
- Left mouse drag: rotate manually

## Next steps

1. Validate that 359 degrees transitions cleanly back to 0 degrees.
2. Benchmark orbit inference speed.
3. Add perceptual and edge-aware losses if L1/MSE produces blur.
4. Add a second tree and a tree-identity latent.
5. Add depth or alpha prediction.
6. Add persistent object memory and progressive refinement.

## Current limitations

The orbit model currently learns one fixed tree and one fixed lighting setup. It is a compact neural renderer, not yet a general tree generator or a 3D world model.
