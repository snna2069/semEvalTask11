"""
Test script for evaluation module.
Creates mock data and runs all evaluation metrics.
"""

import json
import jsonlines
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import run_evaluation


def create_mock_data():
    """Create mock gold labels and predictions for testing."""
    
    # Create data directory
    data_dir = Path("data")
    results_dir = Path("results")
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Mock gold labels with various fields
    gold_data = [
        {
            "id": "en_001",
            "sentence": "The cat sat on the mat.",
            "label": 1,  # plausible
            "language": "en",
            "content_effect": 0,
            "content_id": "content_001"
        },
        {
            "id": "es_001",
            "sentence": "El gato se sentó en la alfombra.",
            "label": 1,  # plausible
            "language": "es",
            "content_effect": 0,
            "content_id": "content_001"  # same content as en_001
        },
        {
            "id": "en_002",
            "sentence": "The building fell upwards into the sky.",
            "label": 0,  # implausible
            "language": "en",
            "content_effect": 1,
            "content_id": "content_002"
        },
        {
            "id": "es_002",
            "sentence": "El edificio cayó hacia arriba en el cielo.",
            "label": 0,  # implausible
            "language": "es",
            "content_effect": 1,
            "content_id": "content_002"  # same content as en_002
        },
        {
            "id": "en_003",
            "sentence": "She drank a glass of water.",
            "label": 1,  # plausible
            "language": "en",
            "content_effect": 0,
            "content_id": "content_003"
        },
        {
            "id": "en_004",
            "sentence": "He ate the liquid book with a fork.",
            "label": 0,  # implausible
            "language": "en",
            "content_effect": 1,
            "content_id": "content_004"
        },
        {
            "id": "fr_001",
            "sentence": "Le chat s'est assis sur le tapis.",
            "label": 1,  # plausible
            "language": "fr",
            "content_effect": 0,
            "content_id": "content_001"  # same content as en_001
        },
        {
            "id": "en_005",
            "sentence": "The sun rises in the east.",
            "label": 1,  # plausible
            "language": "en",
            "content_effect": 0,
            "content_id": "content_005"
        },
    ]
    
    # Mock predictions (some correct, some incorrect)
    predictions = [
        {"id": "en_001", "label": 1, "confidence": 0.95},  # correct
        {"id": "es_001", "label": 1, "confidence": 0.92},  # correct
        {"id": "en_002", "label": 0, "confidence": 0.88},  # correct
        {"id": "es_002", "label": 0, "confidence": 0.85},  # correct
        {"id": "en_003", "label": 1, "confidence": 0.78},  # correct
        {"id": "en_004", "label": 1, "confidence": 0.65},  # incorrect (should be 0)
        {"id": "fr_001", "label": 0, "confidence": 0.72},  # incorrect (should be 1)
        {"id": "en_005", "label": 1, "confidence": 0.93},  # correct
    ]
    
    # Save gold labels
    gold_file = data_dir / "gold_test.jsonl"
    with jsonlines.open(gold_file, mode='w') as writer:
        for item in gold_data:
            writer.write(item)
    
    print(f"Created gold labels: {gold_file}")
    print(f"  Total examples: {len(gold_data)}")
    print(f"  Languages: {set(item['language'] for item in gold_data)}")
    
    # Save predictions
    pred_file = results_dir / "predictions_test.jsonl"
    with jsonlines.open(pred_file, mode='w') as writer:
        for item in predictions:
            writer.write(item)
    
    print(f"\nCreated predictions: {pred_file}")
    print(f"  Total predictions: {len(predictions)}")
    
    return str(pred_file), str(gold_file)


def main():
    """Run evaluation test."""
    print("="*60)
    print("EVALUATION MODULE TEST")
    print("="*60)
    print()
    
    # Create mock data
    print("Step 1: Creating mock data...")
    print("-"*60)
    pred_file, gold_file = create_mock_data()
    
    # Run evaluation
    print("\n" + "="*60)
    print("Step 2: Running evaluation...")
    print("-"*60)
    results = run_evaluation(
        pred_file=pred_file,
        gold_file=gold_file,
        output_dir="results"
    )
    
    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. Check that all metrics are calculated correctly")
    print("3. Get the trained model")
    print("4. Run inference on real data using src/inference.py")


if __name__ == "__main__":
    main()