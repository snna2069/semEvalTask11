"""
SemEval 2026 Task 11 - Inference Module

This module handles model loading and inference for plausibility prediction.
Supports both single predictions and batch processing of JSONL files.
"""

import torch
import jsonlines
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import json

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")


class PlausibilityPredictor:
    """
    Wrapper class for plausibility prediction model.
    
    Handles model loading, tokenization, and inference.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize predictor with model and tokenizer.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            max_length: Maximum sequence length for tokenization
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_path = Path(model_path)
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer from checkpoint."""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Try to load config first
            config = AutoConfig.from_pretrained(self.model_path)
            print(f"Model architecture: {config.model_type}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                config=config
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTroubleshooting tips:")
            print("1. Ensure model files exist in the specified directory")
            print("2. Check that the model was saved with save_pretrained()")
            print("3. Verify the model is compatible with transformers library")
            raise
    
    def predict(
        self,
        text: str,
        return_confidence: bool = True
    ) -> Union[int, Dict[str, Union[int, float]]]:
        """
        Predict plausibility for a single text.
        
        Args:
            text: Input text to classify
            return_confidence: If True, return dict with label and confidence
            
        Returns:
            If return_confidence=False: integer label (0 or 1)
            If return_confidence=True: dict with 'label' and 'confidence'
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get prediction
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label].item()
        
        if return_confidence:
            return {
                'label': pred_label,
                'confidence': confidence
            }
        else:
            return pred_label
    
    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Predict plausibility for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of dicts with 'label' and 'confidence' for each text
        """
        results = []
        
        # Create batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Running inference")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get predictions
                pred_labels = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1).values
            
            # Collect results
            for label, conf in zip(pred_labels.cpu().numpy(), confidences.cpu().numpy()):
                results.append({
                    'label': int(label),
                    'confidence': float(conf)
                })
        
        return results


def process_jsonl_file(
    input_file: str,
    model_path: str,
    output_file: str,
    batch_size: int = 32,
    device: Optional[str] = None,
    text_field: str = 'sentence'
) -> None:
    """
    Process an entire JSONL file and generate predictions.
    
    Expected input format per line:
    {
        "id": "unique_identifier",
        "sentence": "text to classify",
        ... (other fields will be ignored)
    }
    
    Output format per line:
    {
        "id": "unique_identifier",
        "label": 0 or 1,
        "confidence": 0.0-1.0
    }
    
    Args:
        input_file: Path to input JSONL file
        model_path: Path to saved model checkpoint
        output_file: Path to output JSONL file for predictions
        batch_size: Batch size for inference
        device: Device to run inference on
        text_field: Name of the field containing text (default: 'sentence')
    """
    print(f"Processing file: {input_file}")
    
    # Load predictor
    predictor = PlausibilityPredictor(model_path, device=device)
    
    # Read input data
    data = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            data.append(obj)
    
    print(f"Loaded {len(data)} examples")
    
    # Extract texts and IDs
    texts = [item[text_field] for item in data]
    ids = [item['id'] for item in data]
    
    # Run batch prediction
    print("Running inference...")
    predictions = predictor.batch_predict(texts, batch_size=batch_size)
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write predictions
    print(f"Writing predictions to: {output_file}")
    with jsonlines.open(output_file, mode='w') as writer:
        for id_, pred in zip(ids, predictions):
            writer.write({
                'id': id_,
                'label': pred['label'],
                'confidence': pred['confidence']
            })
    
    print(f"Successfully wrote {len(predictions)} predictions")
    
    # Print summary statistics
    num_plausible = sum(1 for p in predictions if p['label'] == 1)
    num_implausible = len(predictions) - num_plausible
    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
    
    print("\nPrediction Summary:")
    print(f"  Plausible (label=1): {num_plausible} ({num_plausible/len(predictions)*100:.1f}%)")
    print(f"  Implausible (label=0): {num_implausible} ({num_implausible/len(predictions)*100:.1f}%)")
    print(f"  Average confidence: {avg_confidence:.4f}")


def test_inference_with_mock_data():
    """
    Test inference pipeline with mock data.
    Useful for testing before real model is available.
    """
    print("Running mock inference test...")
    
    # Create mock input file
    mock_input = Path("data/test_sample.jsonl")
    mock_input.parent.mkdir(parents=True, exist_ok=True)
    
    mock_data = [
        {"id": "test_001", "sentence": "The cat sat on the mat.", "language": "en"},
        {"id": "test_002", "sentence": "The building fell upwards into the sky.", "language": "en"},
        {"id": "test_003", "sentence": "She drank a glass of water.", "language": "en"},
        {"id": "test_004", "sentence": "He ate the liquid book with a fork.", "language": "en"},
    ]
    
    with jsonlines.open(mock_input, mode='w') as writer:
        for item in mock_data:
            writer.write(item)
    
    print(f"Created mock input file: {mock_input}")
    print("Sample data:")
    for item in mock_data:
        print(f"  {item['id']}: {item['sentence']}")
    
    # Create mock predictions (random for testing)
    import random
    mock_output = Path("results/predictions_test.jsonl")
    mock_output.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(mock_output, mode='w') as writer:
        for item in mock_data:
            writer.write({
                'id': item['id'],
                'label': random.choice([0, 1]),
                'confidence': random.uniform(0.6, 0.99)
            })
    
    print(f"\nCreated mock predictions: {mock_output}")
    print("\nMock test completed successfully!")
    print("Next step: Replace with real model inference once model is available.")


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference for SemEval 2026 Task 11 plausibility prediction"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on JSONL file')
    infer_parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model checkpoint'
    )
    infer_parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    infer_parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output JSONL file for predictions'
    )
    infer_parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    infer_parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on (default: auto-detect)'
    )
    infer_parser.add_argument(
        '--text_field',
        type=str,
        default='sentence',
        help='Name of field containing text (default: sentence)'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run mock inference test')
    
    args = parser.parse_args()
    
    if args.command == 'infer':
        process_jsonl_file(
            input_file=args.input_file,
            model_path=args.model_path,
            output_file=args.output_file,
            batch_size=args.batch_size,
            device=args.device,
            text_field=args.text_field
        )
    elif args.command == 'test':
        test_inference_with_mock_data()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()