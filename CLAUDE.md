# CLAUDE.md - AnomalyCLIP Project

## Project Overview

This is the official **AnomalyCLIP** implementation (ICLR 2024): Object-agnostic Prompt Learning for Zero-shot Anomaly Detection. The model adapts CLIP for industrial anomaly detection without category-specific training.

**Backbone**: ViT-B/16 (adapted from original ViT-L/14@336px)
**GPU**: RTX 3060 Laptop 6GB
**GitHub**: https://github.com/yibo0210/anomalyclip

## Architecture (Four Innovations)

| # | Component | File | Mechanism |
|---|-----------|------|-----------|
| 1 | **DPAM** | `AnomalyCLIP_lib/AnomalyCLIP.py:Attention` | Replace Q/K with V in deep layer self-attention |
| 2 | **Object-agnostic Prompt** | `prompt_ensemble.py:ctx_pos/ctx_neg` | Learn "object"/"damaged object" semantics |
| 3 | **Compound Prompt** | `prompt_ensemble.py:compound_prompts_text` | Layer-wise learnable tokens in text encoder |
| 4 | **Multi-layer Summation** | `train.py:similarity_map_list` | Per-layer anomaly maps → simple sum |

## Project Structure

```
AnomalyCLIP-main/
├── train.py / test.py           # Training & evaluation entry points
├── prompt_ensemble.py            # Prompt learner (ctx_pos/neg, compound prompts)
├── dataset.py                    # Data loader (MVTec, VisA, BTAD, MPDD)
├── loss.py / metrics.py          # Focal loss, Dice loss, AUROC, AUPRO
├── utils.py / logger.py          # Utilities
├── visualization.py              # Heatmap overlay
│
├── AnomalyCLIP_lib/              # Core model library
│   ├── AnomalyCLIP.py            #   AnomalyCLIP model (DPAM + learnable tokens)
│   ├── CLIP.py                   #   Standard CLIP (ResNet + ViT)
│   ├── build_model.py            #   Model factory from state_dict
│   ├── model_load.py             #   Model loading + similarity computation
│   └── simple_tokenizer.py       #   BPE tokenizer
│
├── generate_dataset_json/        # Dataset JSON generators
│   ├── mvtec.py / visa.py / btad.py / mpdd.py
│
├── scripts/                      # Utility scripts
│   ├── run_demo.py               #   Single image inference demo
│   └── plot_results.py           #   Experiment result plotting
│
├── docs/                         # Documentation
│   ├── literature_survey.md      #   14-paper survey with feasibility analysis
│   └── improvement_plan.md       #   Phase 1-4 improvement plan
│
├── assets/                       # README images
├── checkpoints/                  # Saved model weights (gitignored)
├── data/                         # Datasets (gitignored)
├── clip_models/                  # CLIP pretrained weights (gitignored)
└── results/                      # Evaluation outputs (gitignored)
```

## Experiment Branches

```
main ──────────────────────────────────── baseline (original paper code)
  │
  ├── exp/multiscale    → Phase 1: Multi-Scale Spatial Aggregation
  ├── exp/resolution    → Phase 2: Higher input resolution (224→336)
  └── exp/attn-adapter  → Phase 3: Attention-Focused Adapter
```

## Key Parameters

| Param | ViT-B/16 | ViT-L/14 |
|-------|----------|----------|
| `--features_list` | [3, 6, 9, 12] | [6, 12, 18, 24] |
| `--image_size` | 224 | 518 |
| `--batch_size` | 16 | 8 |
| `--model_name` | ViT-B/16 | ViT-L/14@336px |

## Current Baseline (MVTec AD)

| pixel_auroc | pixel_aupro | image_auroc | image_ap |
|:-----------:|:-----------:|:-----------:|:--------:|
| 94.1 | 84.0 | 90.4 | 94.8 |

Target: pixel_aupro 84 → 90+ (original ViT-L/14: 93.9)

## Behavioral Guidelines

1. **Branch for each experiment** — never modify main directly
2. **Run evaluate after each experiment** — record pixel_auroc, pixel_aupro, image_auroc, image_ap
3. **Compare against baseline** — record delta for each metric
4. **Commit after each completed experiment** — with results in commit message
5. **Stop training early** when loss plateaus for 3+ epochs
6. **Save and record all data** after each experiment
