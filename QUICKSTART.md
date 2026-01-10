# Quick Start Guide

This guide will help you quickly set up and run the model training and evaluation framework.

## Step 1: Install Dependencies

```bash
pip install torch numpy tqdm
```

**Note**: The framework automatically detects and uses GPU if available (e.g., Google Colab). For CPU-only training, it will automatically fall back to CPU.

## Step 2: Prepare Training Data

Option A: Use sample data (for testing):
```bash
python3.11 setup_data.py --data_dir data/ --num_files 50
```

Option B: Use your own TypeScript/JavaScript files:
- Place `.ts`, `.js`, `.tsx`, or `.jsx` files in the `data/` directory
- The framework will automatically collect and tokenize them

## Step 3: Train Models

Train all three models (one at a time):

```bash
# Mamba-2 (auto-detect GPU/CPU)
python3.11 training/train.py --model mamba2 --data_dir data/ --epochs 5 --batch_size 8

# RWKV-X  
python3.11 training/train.py --model rwkv_x --data_dir data/ --epochs 5 --batch_size 8

# xLSTM
python3.11 training/train.py --model xlstm --data_dir data/ --epochs 5 --batch_size 8
```

**Note**: 
- GPU is automatically detected and used if available (e.g., Google Colab)
- For CPU-only: `--device cpu`
- For GPU-only: `--device cuda`
- Training on CPU will be slow. On GPU (Colab), you can use larger batch sizes (e.g., `--batch_size 32`)

Checkpoints will be saved in `checkpoints/` directory.

## Step 4: Evaluate Models

Evaluate a single model:
```bash
python3.11 evaluation/eval.py --checkpoint checkpoints/mamba2_best_model.pt
```

Compare all three models:
```bash
python3.11 compare_models.py \
    --mamba2_checkpoint checkpoints/mamba2_best_model.pt \
    --rwkv_x_checkpoint checkpoints/rwkv_x_best_model.pt \
    --xlstm_checkpoint checkpoints/xlstm_best_model.pt
```

## Step 5: View Results

Results are saved as JSON files in `results/` directory:
- Individual model results: `results/results_<model>_<timestamp>.json`
- Comparison results: `results/model_comparison.json`

## Troubleshooting

### TypeScript Compilation Errors
If you see errors about `tsc` or `bun` not found:
- The framework will fall back to basic syntax checking
- Or install Bun: `curl -fsSL https://bun.sh/install | bash`
- For full evaluation, install TypeScript: `bun install -g typescript`

### Out of Memory
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--d_model` (try 256)
- Reduce `--n_layers` (try 4)
- Reduce `--max_seq_len` (try 256)

### Slow Training
- This is expected on CPU! Training is much slower than GPU
- Use fewer epochs for testing (`--epochs 2`)
- Consider training overnight for full runs

## Architecture Details

### Mamba-2
- State Space Model with Structured State Space Duality (SSD)
- Efficient long-context modeling with linear complexity
- Key parameters: `d_state`, `d_conv`, `expand`

### RWKV-X
- RNN with sparse attention mechanism
- Combines time-mixing and channel-mixing
- Key parameters: `attn_size`, `sparse_topk`

### xLSTM
- Extended LSTM with exponential gating and matrix memory
- Supports both scalar (sLSTM) and matrix (mLSTM) memory variants
- Key parameters: `head_dim`, `use_mlstm`

## Next Steps

1. Experiment with different hyperparameters
2. Add more test cases to the test suites
3. Train on larger datasets
4. Compare results across different model sizes

For detailed information, see the main [README.md](README.md).
