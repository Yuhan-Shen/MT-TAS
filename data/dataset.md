# MEKA Dataset

This repository contains the MEKA dataset for multi-task temporal action segmentation, along with data from the EgoPER dataset for single-task temporal action segmentation.

## Dataset Overview

The dataset includes both single-task temporal action segmentation (TAS) data from EgoPER and our multi-task TAS data from MEKA. All data is processed at **10 FPS** for consistency.

## Download

📥 **Dataset Download**: [Google Drive Link](https://drive.google.com/drive/folders/1dh7l8uk5X0CTZ49fjgtMRotK8yW6THzE)

## File Structure

```
MEKA_dataset/
├── features/           # I3D features for all videos
├── groundTruth/        # Frame-level action annotations
├── splits/             # Train/test split definitions
├── videos/             # MP4 video files (MEKA only)
└── mapping.txt         # Action label mapping
```

### 📁 `features/`
Contains I3D (Inflated 3D ConvNet) features extracted from videos at 10 FPS.

- **Format**: `.npy` files
- **Content**: Pre-extracted visual features for both EgoPER and MEKA datasets
- **Usage**: Input features for temporal action segmentation models
- **Naming**: `{video_id}.npy`

### 📁 `groundTruth/`
Contains frame-level action annotations for temporal action segmentation.

- **Format**: `.txt` files
- **Content**: Frame-by-frame action labels
- **Coverage**: 
  - **EgoPER**: Single-task temporal action segmentation annotations
  - **MEKA**: Multi-task temporal action segmentation annotations
- **Naming**: `{video_id}.txt`
- **Structure**: One action label per line, corresponding to each frame

### 📁 `splits/`
Contains train/test split definitions for reproducible experiments.

- **Format**: `.bundle` files
- **Content**: Lists of video IDs for training and testing

### 📁 `videos/`
Contains original video files for the MEKA dataset.

- **Format**: `.mp4` files
- **Coverage**: MEKA dataset videos only
- **Frame Rate**: 10 FPS
- **Note**: For EgoPER videos, please refer to the [EgoPER website](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm)

### 📄 `mapping.txt`
Action label mapping file that defines the correspondence between action names and numerical labels.

- **Format**: Text file with delimiter-separated values
- **Structure**: `{label_id}|{action_name}`
- **Note**: We merge some action classes from the original single-task action taxnomy in EgoPER to accomodate the multi-task setting.
- 
## Dataset Details

### EgoPER (Single-task TAS)
- **Task**: Single-task temporal action segmentation
- **Domain**: Egocentric procedural videos
- **Website**: [https://www.khoury.northeastern.edu/home/eelhami/egoper.htm](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm)
- **Videos**: Available from original EgoPER website
- **Features & Annotations**: Included in this dataset
- **Note**:  The annotations provided here differ from the original EgoPER annotations. We have merged some action classes from the original single-task action taxonomy and made modifications to accommodate the multi-task setting.

### MEKA (Multi-task TAS)
- **Task**: Multi-task temporal action segmentation
- **Domain**: Egocentric procedural videos with multiple concurrent tasks
- **Videos**: Included in `videos/` folder
- **Features & Annotations**: Included in this dataset

## Future Releases

🚀 **Coming Soon**: We are preparing the release of additional modalities. Please stay tuned for updates!

## Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{Shen:CVPR25,
  title={Understanding Multi-Task Activities from Single-Task Videos},
  author={Shen, Yuhan and Elhamifar, Ehsan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## Contact

For questions, issues, or collaborations, please contact:

📧 **Email**: [shen.yuh@northeastern.edu](mailto:shen.yuh@northeastern.edu)


**Note**: This dataset builds upon the EgoPER dataset. Please also cite the original EgoPER work if you use the EgoPER portion of this dataset.
