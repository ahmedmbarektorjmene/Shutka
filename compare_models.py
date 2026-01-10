"""
Script to compare all three models on the same test suites
"""
import argparse
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EvaluationConfig
from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Compare all three models')
    parser.add_argument('--mamba2_checkpoint', type=str, required=True,
                        help='Path to Mamba-2 checkpoint')
    parser.add_argument('--rwkv_x_checkpoint', type=str, required=True,
                        help='Path to RWKV-X checkpoint')
    parser.add_argument('--xlstm_checkpoint', type=str, required=True,
                        help='Path to xLSTM checkpoint')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    checkpoints = {
        'mamba2': args.mamba2_checkpoint,
        'rwkv_x': args.rwkv_x_checkpoint,
        'xlstm': args.xlstm_checkpoint
    }
    
    all_results = {}
    
    # Evaluate each model
    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*60}")
        
        config = EvaluationConfig()
        config.checkpoint_path = checkpoint_path
        config.results_dir = args.results_dir
        
        evaluator = Evaluator(config)
        results = evaluator.evaluate_all()
        all_results[model_name] = results
    
    # Create comparison report
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Syntax':<10} {'Programming':<12} {'Algorithmic':<12} {'Composite':<10}")
    print("-" * 60)
    
    for model_name, results in all_results.items():
        scores = results['scores']
        comparison['models'][model_name] = {
            'checkpoint': checkpoints[model_name],
            'scores': scores
        }
        
        print(f"{model_name:<15} {scores['syntax']:<10.2%} {scores['programming']:<12.2%} "
              f"{scores['algorithmic']:<12.2%} {scores['composite']:<10.2%}")
    
    print(f"{'='*60}\n")
    
    # Save comparison
    os.makedirs(args.results_dir, exist_ok=True)
    comparison_file = os.path.join(args.results_dir, 'model_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison saved to: {comparison_file}")


if __name__ == '__main__':
    main()
