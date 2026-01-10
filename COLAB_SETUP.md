# Google Colab Setup Guide

This guide will help you set up and run the training framework on Google Colab.

## Step 1: Mount Google Drive (Optional)

If you want to save checkpoints to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2: Install Dependencies

```bash
!pip install torch numpy tqdm
```

## Step 3: Upload Project Files

Upload your project folder to Colab, or clone from GitHub:

```bash
# Option 1: Upload via Colab file browser
# Option 2: Clone from GitHub (if you have a repo)
# !git clone https://github.com/yourusername/junior-agent.git
```

## Step 4: Set Up Data

```bash
# Create data directory
!mkdir -p data

# Option 1: Upload TypeScript files via Colab file browser
# Option 2: Download a dataset
# !wget https://example.com/dataset.zip
# !unzip dataset.zip -d data/
```

Or use the sample data generator:

```bash
!python setup_data.py --data_dir data/ --num_files 200
```

## Step 5: Train Models

The framework will automatically detect and use GPU:

```bash
# Mamba-2
!python training/train.py --model mamba2 --data_dir data/ --epochs 20 --batch_size 32

# RWKV-X
!python training/train.py --model rwkv_x --data_dir data/ --epochs 20 --batch_size 32

# xLSTM
!python training/train.py --model xlstm --data_dir data/ --epochs 20 --batch_size 32
```

**GPU Tips:**
- Use larger batch sizes (32-64) on GPU
- Training will be much faster than CPU
- Monitor GPU memory usage with `!nvidia-smi`

## Step 6: Evaluate Models

```bash
!python evaluation/eval.py --checkpoint checkpoints/mamba2_best_model.pt
```

## Step 7: Download Results

```bash
# Download checkpoints
from google.colab import files
files.download('checkpoints/mamba2_best_model.pt')

# Download results
files.download('results/results_mamba2_best_model_*.json')
```

## Complete Colab Notebook Example

```python
# Cell 1: Install dependencies
!pip install torch numpy tqdm

# Cell 2: Setup (upload files or clone repo)
import os
os.chdir('/content')  # or your project directory

# Cell 3: Generate sample data
!python setup_data.py --data_dir data/ --num_files 200

# Cell 4: Train model
!python training/train.py --model mamba2 --data_dir data/ --epochs 20 --batch_size 32

# Cell 5: Evaluate
!python evaluation/eval.py --checkpoint checkpoints/mamba2_best_model.pt

# Cell 6: Check GPU usage
!nvidia-smi
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--d_model` (try 256)
- Reduce `--n_layers` (try 4)
- Use gradient checkpointing (already enabled)

### GPU Not Detected
- Check Runtime > Change runtime type > Hardware accelerator > GPU
- Verify with: `import torch; print(torch.cuda.is_available())`

### Slow Training
- Ensure GPU is enabled in Colab
- Use larger batch sizes (32-64)
- Check GPU utilization with `nvidia-smi`

## Performance Expectations

- **CPU (Ryzen 3 3200G)**: ~1-2 minutes per epoch
- **GPU (Colab T4)**: ~10-30 seconds per epoch
- **GPU (Colab V100)**: ~5-15 seconds per epoch
