
# Dual Translation Systems (Seq2Seq + Transformer)

An educational/research project that builds a machine translation system with **two approaches**:
1) **Classic**: RNN/LSTM Seq2Seq with Attention  
2) **Modern**: Transformer (Encoder–Decoder)

Goal: compare translation quality, training speed, and implementation complexity.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results & Reporting](#results--reporting)
- [Team Practices](#team-practices)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Seq2Seq + Attention** (Bahdanau/Luong) with LSTM
- **Transformer** baseline with shared training pipeline
- Tokenization with BPE/SentencePiece (configurable)
- Reproducible experiments via YAML configs and seeds
- Metrics: **BLEU** (+ optional SacreBLEU), length/coverage stats
- GPU/CPU support, progress logging, checkpoints & early stopping

---

## Requirements
- Python **3.10+**
- Git (and optionally **Git LFS** for large artifacts)
- (Optional) NVIDIA driver/CUDA for GPU training

### Quick PyTorch install
**Conda (recommended):**
```bash
conda create -n nmt python=3.10 -y
conda activate nmt
# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
# or GPU (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
