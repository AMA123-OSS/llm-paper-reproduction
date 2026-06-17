# YUV Reversible Attack Reproduction

This directory contains the core code and curated experiment records for reproducing:

`Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`

## Contents

- `reversible_attack_OURS.py`: original main attack script.
- `atk.py`, `EnModel.py`, `embed_utils.py`, `utils.py`: core attack, ensemble, embedding, and color conversion code.
- `pytorch_grad_cam/`: local CAM implementation used by the original code.
- `runs/`: reproduction scripts, staged notes, CSV metrics, and final summaries.
- `代码归类与功能说明.md`: Chinese guide explaining each uploaded code file's role and function.

## Reproduction Notes

Large runtime artifacts are intentionally not committed:

- virtual environments
- model weights
- `ORI_IMG/` ImageNet samples
- generated RAE PNG outputs
- wheelhouse/cache files

The committed CSV/MD files under `runs/` preserve the reported small-scale reproduction results and recovery-feasibility checks.
