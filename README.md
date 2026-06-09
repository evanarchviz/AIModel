# AIModel

Experimental real-time generative world project.

## Current milestone: Tree GAN v0

The first experiment trains a compact 128×128 DCGAN to generate isolated trees on black backgrounds. This milestone validates the basic dataset, CUDA training, checkpointing, and low-cost image generation pipeline before adding camera-angle conditioning, spatial memory, depth, or world coherence.

### Current result

- 65 cleaned RGB training images
- 16 validation images
- 128×128 output
- black background
- RTX 4070 Super training
- roughly 20 epochs per 8 seconds in the current setup
- recognizably different conifer and broadleaf outputs by epoch 500

The dataset masks are stored beside the RGB files, so the loader explicitly excludes filenames ending in `_mask.png`.

## Setup

Use Python 3.11 or 3.12.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build, then install the remaining requirements:

```powershell
pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Train

```powershell
python train.py --data "C:\path\to\single_tree_dataset_v3\train" --epochs 1000 --batch-size 16
```

The loader should report:

```text
Loaded 65 RGB training images (mask files excluded)
```

Samples are written to `runs/tree_gan/samples/` and the latest checkpoint to `runs/tree_gan/checkpoints/latest.pt`.

## Generate

```powershell
python generate.py --checkpoint "runs/tree_gan/checkpoints/latest.pt" --out generated_trees.png
```

## Next steps

1. Replace transposed convolutions with upsample + convolution blocks.
2. Add explicit alpha/mask prediction.
3. Benchmark inference latency.
4. Add latent interpolation tests.
5. Add camera-angle conditioning.
6. Train on multi-view samples of the same tree.
7. Add compact object memory and progressive refinement.

## Current limitations

This model generates independent tree images. It does not yet understand rotation, depth, persistent identity, or novel-view consistency.
