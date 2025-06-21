# MT-TAS Code Documentation

This directory contains the implementation of **Multi-Task Temporal Action Segmentation (MT-TAS)** for understanding multi-task activities from single-task videos.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision numpy
```

### Download Data

1. Download the MEKA dataset following instructions in [`../data/dataset.md`](../data/dataset.md)
2. Organize your data as:
```
data/
├── [dataset_name]/
│   ├── features/         # I3D features (.npy files)
│   ├── groundTruth/      # Frame-level annotations (.txt files)
│   ├── splits/           # Train/test splits (.bundle files)
│   └── mapping.txt       # Action label mapping
```

### Run Training

```bash
# Basic training
python main.py --action train --dataset meka --split 1


```

## ⚙️ Key Arguments

### Basic Settings
- `--action`: `train` or `predict`
- `--dataset`: Dataset name
- `--split`: Dataset split number
- `--num_epochs`: Training epochs
- `--lr`: Learning rate 

### MT-TAS Components
- `--use_sbl`: Enable Segment Boundary Learning
- `--use_fbfc`: Enable Foreground-Background Feature Composition
- `--no_faar`: Disable Foreground-Aware Refinement (enabled by default)
- `--dual_use_att`: Use attention in dual encoder
- `--features_path`: Path to feature files
- `--fg_bg_features_path`: Path to foreground/background features

## 📊 Output

Results are saved to:
- **Models**: `./models/[exp_id]/[dataset]/split_[X]/`
- **Predictions**: `./results/[exp_id]/[dataset]/epoch[Y]/split_[X]/`
- **Evaluation**: `split[X].eval.json` with accuracy and F1 scores

## 📞 Support

For questions: [shen.yuh@northeastern.edu](mailto:shen.yuh@northeastern.edu)
