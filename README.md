# RSNA Knee Abnormality Detection — Team HPDA (223rd / 6,299)

Code of team **HPDA thì ngon luôn** (dunguit, giaanp, hinnguync, mphatfromuit, nguynxunphc662) for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition.
Best entry: **0.959 public / 0.944 private**. The full write-up is in [reports/HPDA_solution_writeup.md](reports/HPDA_solution_writeup.md).

This repo only holds **our own scripts and notebooks**. Public models and code we built on are cited below and are
not copied here. No weights, labels or data are included.

## What we built

- **Model B (EfficientNetV2-S)**, `tf_efficientnetv2_s.in21k_ft_in1k`. It reads knee-centred 2.5D views (px256 v4 cache,
  crop 224, 3 slices, 3.5 mm, spatial attention, arm `E_prior`, 10 hash folds, 16 epochs) and is trained on 4,349
  report-only studies with distilled soft labels. Solo public score 0.950.
- **ViT-S (DINOv2)**, `vit_small_patch14_dinov2.lvd142m`. Same trainer and recipe as B, run with `--backbone` overridden.
- **Specialist blending.** B takes 25–62.5 % of the Synovitis column, and dkim54 v11i takes 25–50 % of Fracture.
  These shares were fixed before each submission and never fitted on the LB.
- **Evaluation protocol.** We decided changes on gold-58 and ref392 (ref-A/ref-B split), not on the public LB.

## Layout

```
scripts/                     generators/launchers (Python) that build the Kaggle notebooks & kernels
  build_b_px_shards.py         px256 v4 pixel-cache shards (CPU kernels)
  build_b_train.py             B training as a Kaggle T4 kernel (fp16, resumable)
  modal_btrain.py, spawn_btrain.py   B / ViT-S training on Modal A100s, one GPU per fold
  modal_to_kaggle.py, modal_getfile.py   move fold weights Modal -> Kaggle dataset / local
  build_b_solo.py, build_b_subs.py, build_b_tta.py, build_b_ref450.py, build_b_gold58.py ...  B eval/submission notebooks
  build_g*_variants.py, build_h_variants.py, run_g_variants.py   blend submission variants (G6R/G6RM/G6RW ...)
  build_dk_*.py                dkim54 v11i Fracture specialist (eval / subs)
  build_r6_*.py, build_gold_components.py   OOF / gold-58 component scoring
train_kit/
  b_train/compact_train_shards.py   B trainer (our extension of heliosli's compact_train: multi-shard --px, fp16 on T4,
                                   ViT/LLRD, syn-decomp, pair ranking, ... see the header changelog)
  kaggle_train/                Raptor-geometry training kernels (early ConvNeXt / CoAtNet / EffV2-S runs)
  labels/api_labeler.py        LLM second-reader for reports
  tools/                       label peeks, gold-58 blend scoring
notebooks/
  submissions/                 milestone LB submissions (0956 -> G6R -> G6RM/G6RW -> +B4 Synovitis -> +ViT-S)
  experiments/                 B solo/TTA/ref450/gold-58 evals, dk specialist, zoom, V-JEPA, training kernels
reports/                       solution write-up, figures, model comparison
```

Progression (public / private): 0.956/0.940 (public recipe) → 0.957/0.942 (G6R, s6fall third leg) → 0.958/0.943
(G6RM/G6RW) → 0.958/0.944 (+B4 Synovitis 25 %, dk Fracture 25 %) → **0.959/0.944** (B4 Synovitis 50 %).

## Credits (public work used, not included here)

All weights and recipes below belong to their authors.

- **nartaa**: Raptor CoAtNet-384/224 SWA weights (`rsna-knee-publication-swa-weights-20261007`), s6fall
  (`rsna-knee-s6fall-weights`), and the Raptor inference script.
- **heliosli**: R6/R7 ConvNeXt-S readers and stackers, the compact study pipeline (compact_train / compact_infer /
  px_cache / native_cache / anchor bank) we trained B with, R6 OOF predictions, per-finding Raptor weights, and
  OrthoFoundation weights.
- **dkim54**: v11i models, used on Fracture (`rsna-knee-models-v11i`).
- **lingyu07**: v53 master labels (CC0).
- **zoliseress**: reference labels, used for evaluation (and for ref-A training from 10-19).
- **stevenleehans**, **pilkwang**: report label sets used in our early label versions.
- **TranBaDat2607**: the s6fall third-leg recipe.
- Also consulted: [TianK003/RSNA-KneeMRI-kaggle-competition](https://github.com/TianK003/RSNA-KneeMRI-kaggle-competition),
  [chrishotza/RSNA](https://github.com/chrishotza/RSNA),
  [VihaanGopee/rsna-knee-abnormality-detection](https://github.com/VihaanGopee/rsna-knee-abnormality-detection).

The notebooks in `notebooks/` are the exact files we pushed to Kaggle. Several of them embed the public inference code
listed above verbatim, which is how a Kaggle offline submission has to run.
