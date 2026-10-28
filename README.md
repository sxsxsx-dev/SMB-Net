


Multi-modal brain tumor segmentation on BraTS (FLAIR / T1ce / T1 / T2) with modality-missing support, combining a 3D Swin Transformer encoder with Mamba (SSM) blocks for long-range feature modeling.

## Architecture

- **Swin Transformer encoder** (3D window attention) extracts multi-scale features for each modality independently.
- **Mamba blocks** (bidirectional SSM, `mamba_ssm`) further encode/decode features at the bottleneck.
- **MaskModal** handles arbitrary missing modalities via binary masks at training/testing time.
- **Decoder_fuse** progressively fuses multi-scale features with deep supervision.

## Project Structure

```
├── model_swinTransformer_mamba.py          # Main model (Swin encoder + Mamba + fusion decoder)
├── model_swinTransformer_mamba_edges.py    # Variant with boundary/edge-aware heads
├── model_swint_mamba_edges_refined.py      # Refined edge variant
├── layers.py                               # Conv / fusion building blocks
├── train.py                                # Main training script (BraTS)
├── train_edges.py / train_edges_refined.py # Edge-aware training variants
├── predict.py                              # Inference
├── predict_vis.py / predict_vis_refined.py # Inference with visualization
├── data/                                   # Datasets, transforms, samplers
└── utils/                                  # Losses, LR scheduler, arg parser
```

## Requirements

- Python 3.10, PyTorch (CUDA)
- [mamba_ssm](https://github.com/state-spaces/mamba) (requires CUDA compilation)
- MONAI, timm, einops, nibabel, TensorBoard

Install:

```bash
pip install torch monai timm einops nibabel tensorboard
pip install mamba-ssm
```

## Usage

Training ( BraTS 2018, 128^3 crops, all 15 modality-missing masks ):

```bash
python train.py --datapath <path/to/BRATS2018_Training_none_npy> --savepath ./out/swintransformer-mamba
```

Inference:

```bash
python predict.py --datapath <path/to/test_data> --resume <checkpoint.pth>
```

Visualization:

```bash
python predict_vis.py --datapath <path/to/test_data> --resume <checkpoint.pth>
```

## Notes

- Input modalities order: FLAIR, T1ce, T1, T2. A 4-dim boolean mask specifies available modalities.
- Data should be preprocessed into `.npy` (see `utils/generate.py`).
- `sampler.py` contains a `np.random.RandomSatate` typo that is kept as-is; fix before use if that path is exercised.
