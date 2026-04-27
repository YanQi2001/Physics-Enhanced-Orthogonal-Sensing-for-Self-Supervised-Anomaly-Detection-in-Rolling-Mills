# Physics-Enhanced Orthogonal Sensing for Self-Supervised Anomaly Detection in Rolling Mills

This repository contains the core dual-branch source code for self-supervised anomaly detection in rolling mills. The implementation combines a CSD Transformer branch for vibration-frequency coupling and a VQ-VAE/TIMAE-style temporal branch for pressure-condition modeling.

## Repository Structure

- `models/`: neural network modules, including the CSD Transformer, VQ-VAE/TIMAE branch, SPD components, and fusion/gating modules.
- `trainers/`: pretraining and joint-training loops for the model branches.
- `losses/`: reconstruction, physics-guided, and Riemannian/SPD-related losses.
- `data/`: dataset loaders, preprocessing utilities, augmentation, WPD expansion, and CSD construction code.
- `configs/`: example configuration files.
- `main.py`: main training entry point.

## What Is Not Included

To keep the public repository clean and reproducible without exposing industrial data, the following files are intentionally excluded:

- raw data and preprocessed `.pt` datasets
- sample CSV/NumPy data files
- model checkpoints and trained weights
- logs, outputs, figures, and experiment result folders
- `inference/` utilities

## Core Preprocessing Settings

The default preprocessing pipeline uses:

- sampling rate: `100 Hz`
- window length: `1024`
- wavelet packet decomposition depth: `L=3`
- selected sub-bands per channel: `K=3`
- CSD construction via Welch's method

See `data/preprocessing.py` and `configs/shaogang.yaml` for implementation details.

## Notes

This release is intended for code review and reproducibility of the core algorithmic framework. Users should prepare their own data in the expected channel order and update paths in the configuration files before training.
