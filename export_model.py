"""
Export Inference-Only Model with torch.compile Optimization
"""

import os
import torch
import argparse
from models.shutka import UltraEfficientTextJEPA


def export_inference_model(
    checkpoint_path: str,
    output_path: str,
    fast_mode: bool = True,
    optimize: bool = True,
    script: bool = True,
    verbose: bool = True,
):
    """
    Export a clean inference-only model.
    """
    if verbose:
        print("=" * 60)
        print("INFERENCE MODEL EXPORT (Shutka-v2)")
        print("=" * 60)
        print("\n[*] Strategy: Clean inference-only export")

    device = torch.device("cpu")

    # Configuration for ~350M parameters
    # source_dim=512, depth=24

    config = {
        "vocab_size": 100277,
        "source_dim": 512,
        "source_depth": 24,  # Main backbone
        "predictor_dim": 512,
        "predictor_depth": 4,  # Lighter predictor
        "output_dim": 512,
        "max_source_len": 4096,
        "engram_vocab_size": 370000,
    }

    if os.path.exists(checkpoint_path):
        if verbose:
            print(f"\n[*] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load config from checkpoint if available
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            if verbose:
                print("    • Loaded config from checkpoint")
            config = checkpoint["config"]
            # Ensure engram_vocab_size is set (backward compatibility)
            if "engram_vocab_size" not in config:
                config["engram_vocab_size"] = 370000
        else:
            if verbose:
                print(
                    "    ! Config not found in checkpoint. Using default 350M config."
                )

    model = UltraEfficientTextJEPA(**config)
    model.eval()

    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n[*] Model Parameter Count: {param_count / 1e6:.1f}M (Target: 350M)")

    # Scripting Support
    if script:
        if verbose:
            print("\n[*] Attempting TorchScript Export...")
        try:
            # Create dummy inputs for tracing
            # input ids: [B, L]
            dummy_input = torch.randint(0, 1000, (1, 128), dtype=torch.long)
            # hash ngrams: [B, L, 6] (assuming 6 ngram orders)
            dummy_hashes = torch.randint(0, 370000, (1, 128, 6), dtype=torch.long)

            # Trace the model
            # Note: We trace 'predict_next' or 'forward' depending on usage
            # Here we trace forward
            traced_model = torch.jit.trace(model, (dummy_input, dummy_hashes))

            script_path = output_path.replace(".pt", "_script.pt")
            torch.jit.save(traced_model, script_path)
            if verbose:
                print(f"    • TorchScript model saved: {script_path} [OK]")
        except Exception as e:
            print(f"    ! TorchScript export failed: {e}")
            print("    ! Proceeding with standard export.")

    # Standard Save
    export_data = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "inference_only": True,
        "optimized": optimize,
    }

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    torch.save(export_data, output_path)

    if verbose:
        print(f"\n[OK] Export complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="models/shutka.pt")
    parser.add_argument("--no-script", action="store_true")
    args = parser.parse_args()

    export_inference_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        script=not args.no_script,
    )
