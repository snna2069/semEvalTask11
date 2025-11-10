"""
Test script for inference module.
Tests the inference pipeline with mock data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import test_inference_with_mock_data


def main():
    """Run inference module test."""
    print("="*60)
    print("INFERENCE MODULE TEST")
    print("="*60)
    print()
    
    print("Testing inference pipeline with mock data...")
    print("(This test doesn't require a trained model)")
    print()
    
    test_inference_with_mock_data()
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Get the baseline model")
    print("2. Place the model in models/baseline_en.pt")
    print("3. Run real inference:")
    print("   python src/inference.py infer \\")
    print("     --model_path models/baseline_en.pt \\")
    print("     --input_file data/test_en.jsonl \\")
    print("     --output_file results/predictions_baseline_en.jsonl")


if __name__ == "__main__":
    main()