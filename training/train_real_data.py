"""
Train Shutka on REAL Data Only (No LLM-generated content)

Streams from:
1. GitHub commits (commit message → code)
2. JSDoc/TSDoc → function implementations
3. Stack Overflow Q&A
4. The Stack (TypeScript/JavaScript code)

ALL STREAMING - no full downloads!
NO hardcoded examples!
NO LLM-generated data!

Usage:
    python train_real_data.py --code_samples 10000 --instruction_samples 2000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shutka import UltraEfficientTextJEPA
from training.real_instruction_loader import CombinedRealDataset
from training.trainer import Trainer
from config import TrainingConfig
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train on REAL data only")

    # Data
    parser.add_argument(
        "--code_samples",
        type=int,
        default=10000,
        help="Code pattern samples from The Stack",
    )
    parser.add_argument(
        "--instruction_samples",
        type=int,
        default=2000,
        help="Instruction samples (commits, docstrings, SO)",
    )

    # Model (Shutka-v2 350M Default)
    parser.add_argument("--source_dim", type=int, default=512)
    parser.add_argument("--max_source_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=100277)
    parser.add_argument("--engram_vocab_size", type=int, default=370000)

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # Optimization
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SHUTKA TRAINING - REAL DATA ONLY")
    print("=" * 60)
    print("\nData sources (all streaming, all real):")
    print("  ✓ The Stack: TypeScript/JavaScript code")
    print("  ✓ GitHub commits: commit message → code change")
    print("  ✓ JSDoc/TSDoc: documentation → implementation")
    print("  ✓ Stack Overflow: real Q&A (filtered for JS/TS)")

    print("\n" + "-" * 40)
    print("Streaming and buffering training data...")
    print("-" * 40)

    # Use CombinedRealDataset from the new loader
    streaming_ds = CombinedRealDataset(
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        max_code_samples=args.code_samples,
        max_instruction_samples=args.instruction_samples,
    )

    # Buffer samples for repeated epochs (since it's streaming, we can't easily seek back)
    print("\nBuffering samples...")
    buffered_samples = []
    try:
        for sample in streaming_ds:
            buffered_samples.append(sample)
            if len(buffered_samples) % 1000 == 0:
                print(f"  Buffered {len(buffered_samples)} samples...")
    except Exception as e:
        print(f"Streaming warning: {e}")

    print(f"\nTotal buffered: {len(buffered_samples)} samples")

    if len(buffered_samples) == 0:
        print(
            "ERROR: No samples loaded! Check your internet connection or dataset availability."
        )
        return

    # Create Dataset from buffered samples
    class BufferedDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    full_dataset = BufferedDataset(buffered_samples)

    # Split train/val
    train_size = int(len(full_dataset) * 0.9)
    val_size = len(full_dataset) - train_size
    # Handle edge case of small dataset
    if val_size == 0 and len(full_dataset) > 1:
        val_size = 1
        train_size = len(full_dataset) - 1
    elif len(full_dataset) == 1:
        train_ds = full_dataset
        val_ds = full_dataset  # Overfit on single sample
    else:
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Collate function (MUST match the one in real_instruction_loader.py to handle hashes)
    def collate_fn(batch):
        source_seqs = [item["source_patches"] for item in batch]
        target_seqs = [item["target_patches"] for item in batch]

        # Handle hash n-grams if present (from Instruction dataset) or pass None
        hashes = [item.get("hash_ngrams", None) for item in batch]

        source_batch = torch.nn.utils.rnn.pad_sequence(
            source_seqs, batch_first=True, padding_value=0
        )
        target_batch = torch.nn.utils.rnn.pad_sequence(
            target_seqs, batch_first=True, padding_value=0
        )

        if source_batch.shape[1] > args.max_source_len:
            source_batch = source_batch[:, : args.max_source_len]
        if target_batch.shape[1] > args.max_target_len:
            target_batch = target_batch[:, : args.max_target_len]

        # Pad Hashes (Batch, Seq, NGrams)
        padded_hashes = None
        if any(h is not None for h in hashes):  # Changed to any, and handle None
            max_len = source_batch.shape[1]
            # Get n_gram_counts from first non-None hash
            first_valid_hash = next((h for h in hashes if h is not None), None)
            if first_valid_hash is not None:
                n_gram_counts = first_valid_hash.shape[1]
                padded_hashes = torch.zeros(
                    (len(batch), max_len, n_gram_counts), dtype=torch.long
                )
                for i, h in enumerate(hashes):
                    if h is not None:
                        seq_len = min(h.shape[0], max_len)
                        padded_hashes[i, :seq_len, :] = h[:seq_len]

        return {
            "source_patches": source_batch,
            "target_patches": target_batch,
            "hash_ngrams": padded_hashes,  # Crucial for Engram Layer
        }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model (Shutka-v2 350M spec)
    print("\n" + "-" * 40)
    print("Initializing model...")
    print("-" * 40)

    model = UltraEfficientTextJEPA(
        vocab_size=args.vocab_size,
        source_dim=args.source_dim,
        source_depth=24,  # 350M standard
        predictor_dim=args.source_dim,
        # predictor_depth=4, # Default in model is 6, can be adjusted
        output_dim=args.source_dim,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        engram_vocab_size=args.engram_vocab_size,
    )

    # Config
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.checkpoint_dir = args.checkpoint_dir
    config.save_every = 1000
    config.eval_every = 500
    # Add rank for GaLore/Custom Optimizer
    config.rank = 128

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        use_compile=not args.no_compile,
        use_mixed_precision=not args.no_mixed_precision,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "=" * 60)
    print(f"STARTING TRAINING ON {trainer.device}")
    print("=" * 60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving...")
        trainer.save_checkpoint("interrupted.pt")
    except Exception as e:
        print(f"\n\nTraining Error: {e}")
        # Try to save anyway
        trainer.save_checkpoint("error_checkpoint.pt")
        raise e

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model: {args.checkpoint_dir}/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
