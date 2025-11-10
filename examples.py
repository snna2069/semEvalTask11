"""
Example usage of inference and evaluation modules.
This demonstrates how to use the modules programmatically.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import PlausibilityPredictor, process_jsonl_file
from src.evaluate import run_evaluation


def example_single_prediction():
    """Example: Single text prediction."""
    print("="*60)
    print("Example 1: Single Prediction")
    print("="*60)
    
    # Initialize predictor (will fail without model)
    # predictor = PlausibilityPredictor('models/baseline_en.pt')
    
    # Example sentences
    sentences = [
        "The cat sat on the mat.",
        "The building fell upwards into the sky.",
        "She drank a glass of water.",
        "He ate the liquid book with a fork."
    ]
    
    print("\nExample sentences to classify:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")
        # result = predictor.predict(sent)
        # print(f"   → Label: {result['label']}, Confidence: {result['confidence']:.3f}")
    
    print("\n(Uncomment predictor code once model is available)")


def example_batch_prediction():
    """Example: Batch prediction from file."""
    print("\n" + "="*60)
    print("Example 2: Batch Prediction")
    print("="*60)
    
    print("\nProcessing JSONL file...")
    print("Command:")
    print("""
    process_jsonl_file(
        input_file='data/test_en.jsonl',
        model_path='models/baseline_en.pt',
        output_file='results/predictions_baseline_en.jsonl',
        batch_size=32
    )
    """)
    
    print("\n(Will run once model is available)")


def example_evaluation():
    """Example: Run evaluation."""
    print("\n" + "="*60)
    print("Example 3: Evaluation")
    print("="*60)
    
    print("\nRunning evaluation on predictions...")
    print("Command:")
    print("""
    results = run_evaluation(
        pred_file='results/predictions_baseline_en.jsonl',
        gold_file='data/gold_en.jsonl',
        output_dir='results/en'
    )
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"TCE: {results['tce_metrics']['tce']:.4f}")
    """)
    
    print("\n(Run this after inference completes)")


def example_programmatic_usage():
    """Example: Complete programmatic workflow."""
    print("\n" + "="*60)
    print("Example 4: Complete Workflow")
    print("="*60)
    
    print("""
# Step 1: Initialize predictor
from src.inference import PlausibilityPredictor

predictor = PlausibilityPredictor(
    model_path='models/baseline_en.pt',
    device='cuda'  # or 'cpu'
)

# Step 2: Make predictions
texts = ["Sentence 1", "Sentence 2", ...]
results = predictor.batch_predict(texts, batch_size=32)

# Results format:
# [
#   {'label': 1, 'confidence': 0.95},
#   {'label': 0, 'confidence': 0.88},
#   ...
# ]

# Step 3: Save predictions
import jsonlines
with jsonlines.open('predictions.jsonl', 'w') as writer:
    for id_, result in zip(ids, results):
        writer.write({
            'id': id_,
            'label': result['label'],
            'confidence': result['confidence']
        })

# Step 4: Evaluate
from src.evaluate import run_evaluation

metrics = run_evaluation(
    pred_file='predictions.jsonl',
    gold_file='gold.jsonl',
    output_dir='results/'
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
    """)


def example_command_line_usage():
    """Example: CLI usage."""
    print("\n" + "="*60)
    print("Example 5: Command Line Usage")
    print("="*60)
    
    print("""
# Test with mock data (no model needed)
python src/inference.py test

# Run inference
python src/inference.py infer \\
  --model_path models/baseline_en.pt \\
  --input_file data/test_en.jsonl \\
  --output_file results/predictions_baseline_en.jsonl \\
  --batch_size 32 \\
  --device cuda

# Run evaluation
python src/evaluate.py \\
  --predictions results/predictions_baseline_en.jsonl \\
  --gold data/gold_en.jsonl \\
  --output results/en

# View results
cat results/en/metrics.json
    """)


def main():
    """Run all examples."""
    print("SemEval 2026 Task 11 - Usage Examples")
    print()
    
    example_single_prediction()
    example_batch_prediction()
    example_evaluation()
    example_programmatic_usage()
    example_command_line_usage()
    
    print("\n" + "="*60)
    print("For more information, see:")
    print("  - README.md: Complete documentation")
    print("  - QUICKSTART.md: Quick start guide")
    print("  - tests/: Working code examples")
    print("="*60)


if __name__ == "__main__":
    main()