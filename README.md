# MarmoPose

## Introduction
A complete multi-marmoset 3D pose tracking system.

## Features
- 3D pose tracking for multiple marmosets, with a comprehensive processing pipeline.
- Supports low-latency closed-loop experimental control based on the 3D poses of marmosets.
- User-friendly deployment in a typical marmoset family cage.
- Employs a marmoset skeleton model for 3D coordinates optimization.

## Getting Started
### Installation
1. Create python environment and install SLEAP based on https://sleap.ai/installation.html
2. ```pip install mayavi ffmpeg tqdm```

## Usage
- Train new models, or download models from https://cloud.tsinghua.edu.cn/d/74adaa2da1a848d6b984
- **run.py**: offline process
- **run_realtime.py**: real-time process
- Specify project and model directory, then follow the comments
