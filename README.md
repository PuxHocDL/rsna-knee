# RSNA Knee Abnormality Detection — EfficientNetV2-S & ViT-S

Training / inference code for our two "direction B" models in the RSNA Knee Abnormality Detection Kaggle competition:

| Model | timm backbone | Notes |
|---|---|---|
| **B (EffNet)** | `tf_efficientnetv2_s.in21k_ft_in1k` | arm `E_prior`, 10 hash folds, 16 epochs, crop 224, 3 slices, 3.5 mm, spatial attention |
| **ViT-S** | `vit_small_patch14_dinov2.lvd142m` | same script/recipe, `--backbone` override (LLRD supported via `--llrd`) |

Both are trained with the same script, `train/compact_train_shards.py`, on px256 v4 multi-view pixel cache shards.

## Layout

```
data/        pixel-cache building (DICOM -> px256 v4 view shards)
  build_b_px_shards.py   Kaggle CPU kernels that write the 10 px shards
  px_cache.py, native_cache.py, side_common.py
train/
  compact_train_shards.py  main trainer (EffV2-S / ViT / ConvNeXt / ResNet ...)
  modal_btrain.py          train folds in parallel on Modal A100s (one GPU per fold)
  spawn_btrain.py          launcher helper; pass overrides e.g. "--backbone vit_small_patch14_dinov2.lvd142m"
  build_b_train.py         same recipe as a Kaggle T4 kernel (fp16, resumable)
  modal_to_kaggle.py, modal_getfile.py   move fold weights Modal -> Kaggle dataset / local
infer/
  compact_infer.py         test-time worker (fold logits averaged)
  build_b_solo.py, build_b_subs.py, zoom_patch.py   build submission notebooks
submission/
  rsna-g6rw-b10cvit-syn-dkfrac.ipynb   final blend notebook using B10c (EffV2-S) + ViT-S models
```

## Quick start (Modal)

```bash
modal run train/modal_btrain.py::setup --files targets_b_v10c.csv
modal run --detach train/modal_btrain.py::train_many --targets targets_b_v10c.csv --tag v10c --folds 0,1,2,3
```

Model weights and data are not included (Kaggle datasets: `rsna-b10c-models`, `rsna-vits-models`).
Credentials are read from the local machine at runtime and are not stored in this repo.
