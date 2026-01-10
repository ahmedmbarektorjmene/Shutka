# Model Training and Evaluation Framework

This project implements, trains, and evaluates three sequence model architectures:
- **Mamba-2**: State Space Model with Structured State Space Duality (SSD)
- **RWKV-X**: RNN with sparse attention mechanism
- **xLSTM**: Extended LSTM with exponential gating and matrix memory

All models are trained from scratch on the same dataset and evaluated on CPU (AMD Ryzen 3 3200G).

## Project Structure

```
.
├── models/              # Model implementations
│   ├── mamba2.py       # Mamba-2 architecture
│   ├── rwkv_x.py       # RWKV-X architecture
│   └── xlstm.py        # xLSTM architecture
├── tokenizer/          # Tokenization modules
│   └── tokenizer.py    # Custom byte-based/BPE tokenizer
├── training/           # Training scripts
│   ├── train.py        # Main training loop
│   └── trainer.py      # Trainer class
├── evaluation/         # Evaluation scripts
│   ├── test_syntax.py      # Test 1: Syntax correctness
│   ├── test_programming.py # Test 2: Programming correctness
│   ├── test_algorithmic.py # Test 3: Algorithmic thinking
│   └── evaluator.py    # Evaluation metrics and reporting
├── data/               # Data directory
├── checkpoints/        # Model checkpoints
└── results/            # Evaluation results

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up training data (optional - creates sample data if directory is empty):
```bash
python setup_data.py --data_dir data/ --num_files 50
```

## Usage

### Training

Train each model on the same dataset:

```bash
# Train Mamba-2
python training/train.py --model mamba2 --data_dir data/ --epochs 10 --batch_size 8

# Train RWKV-X
python training/train.py --model rwkv_x --data_dir data/ --epochs 10 --batch_size 8

# Train xLSTM
python training/train.py --model xlstm --data_dir data/ --epochs 10 --batch_size 8
```

Training options:
- `--model`: Model architecture (`mamba2`, `rwkv_x`, `xlstm`)
- `--data_dir`: Directory containing TypeScript/JavaScript files
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (default: 8, CPU-friendly)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--d_model`: Model dimension (default: 512)
- `--n_layers`: Number of layers (default: 6)
- `--resume`: Resume from checkpoint path

### Evaluation

Evaluate a single model:

```bash
python evaluation/eval.py --checkpoint checkpoints/mamba2_best_model.pt
```

Compare all three models:

```bash
python compare_models.py \
    --mamba2_checkpoint checkpoints/mamba2_best_model.pt \
    --rwkv_x_checkpoint checkpoints/rwkv_x_best_model.pt \
    --xlstm_checkpoint checkpoints/xlstm_best_model.pt
```

Evaluation options:
- `--checkpoint`: Path to model checkpoint
- `--test_suite_dir`: Directory containing test suites (default: `evaluation/test_suites`)
- `--results_dir`: Directory to save results (default: `results`)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_p`: Top-p sampling parameter (default: 0.95)

## Test Suites

The evaluation framework includes three test categories:

1. **Syntax Correctness**: Tests model's ability to complete TypeScript syntax
2. **Programming Correctness**: Tests model's ability to generate working functions with unit tests
3. **Algorithmic Thinking**: Tests model's ability to implement complex algorithms correctly and efficiently

Results are saved as JSON files in the `results/` directory with detailed metrics for each test case.

## References

- [Mamba-2: Scalable State Space Sequence Modeling](https://www.emergentmind.com/topics/mamba2)
- [RWKV-X: A Linear Complexity Hybrid Language Model](https://arxiv.org/html/2504.21463v2)
- [xLSTM: Extended Long Short-Term Memory](https://openreview.net/forum?id=ARAxPPIAhq)
