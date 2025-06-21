# Understanding Multi-Task Activities from Single-Task Videos  
*CVPR 2025 Highlight · Yuhan Shen · Ehsan Elhamifar*

[**📄 Read the Paper (PDF)**](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_Understanding_Multi-Task_Activities_from_Single-Task_Videos_CVPR_2025_paper.pdf)  
[**🖼️ View the Poster (PDF)**](https://yuhan-shen.github.io/files/cvpr25_mttas_poster.pdf)



## Introduction

We introduce **Multi‑Task Temporal Action Segmentation (MT‑TAS)**, a novel framework designed to tackle the complex scenario of interleaved actions in multi‑task videos using only single‑task training data. By leveraging innovative modules, including **Multi‑task Sequence Blending**, **Segment Boundary Learning**, **Dynamic Isolation of Video Elements**, **Foreground-Background Feature Composition**, and **Foreground‑Aware Action Refinement**, the approach synthesizes multi‑task videos and dynamically isolates foreground/background elements to improve multi-task temporal segmentation. To validate its effectiveness, we present the **MEKA** dataset, a 12‑hour egocentric multi‑task kitchen activity collection, demonstrating that MT‑TAS significantly narrows the performance gap between single‑task training and multi‑task evaluation, achieving state‑of‑the‑art results in complex, real‑world environments


## 📊 Dataset Release

**The MEKA dataset is now available!** 

Please refer to [**`data/dataset.md`**](data/dataset.md) for detailed information.

## 💻 Code Release

**The MT-TAS implementation is now available!**

Please refer to [**`codes/codes.md`**](codes/codes.md) for detailed information.
- Usage examples

## Quick Start

1. **Download the dataset**: Follow instructions in [`data/dataset.md`](data/dataset.md)
2. **Setup the code**: Follow instructions in [`codes/codes.md`](codes/codes.md)
3. **Run experiments**: Use the provided scripts for training and evaluation

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{Shen:CVPR25,
  title={Understanding Multi-Task Activities from Single-Task Videos},
  author={Shen, Yuhan and Elhamifar, Ehsan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## Contact

For questions or issues, please contact:
- **Yuhan Shen**: [shen.yuh@northeastern.edu](mailto:shen.yuh@northeastern.edu)

