# RoboLab3D Depth Benchmark

This repository provides a small evaluation framework to benchmark different monocular depth estimation models on the DRENDS dataset.

It:
- Loads stereo pairs and ground-truth depth maps (with masks).
- Runs a chosen depth model on the images.
- Computes **absolute-scale** and **aligned-scale** depth metrics, plus a **temporal consistency** metric.
- Saves predictions and a `metrics_summary.json` per run.
- Provides an `analysis.py` script to aggregate results and plot comparison figures.

---

## 1. Installation

### Requirements

- Python 3.9+
- PyTorch (with CUDA recommended)
- NumPy, OpenCV, pandas, matplotlib
- tifffile, imageio
- tqdm
- (Optional) HuggingFace `transformers` for some models (e.g. DepthPro, ZoeDepth)

Install dependencies (example):

```bash
pip install torch torchvision torchaudio  # or your preferred CUDA wheel
pip install numpy opencv-python pandas matplotlib tifffile imageio tqdm transformers
