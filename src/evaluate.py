"""
SemEval 2026 Task 11 - Evaluation Module

This module implements evaluation metrics for plausibility prediction:
- Accuracy
- True Content Effect (TCE)
- Intra-lingual Plausibility
- Cross-lingual Plausibility
"""

import json
import jsonlines
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import argparse


def load_predictions(pred_file: str) -> pd.DataFrame:
    """
    Load predictions from JSONL file.
    
    Expected format per line:
    {
        "id": "unique_identifier",
        "label": 0 or 1,
        "confidence": 0.0-1.0 (optional)
    }
    
    Args:
        pred_file: Path to predictions JSONL file
        
    Returns:
        DataFrame with columns: id, label, confidence
    """
    predictions = []
    with jsonlines.open(pred_file) as reader:
        for obj in reader:
            predictions.append({
                'id': obj['id'],
                'label': obj['label'],
                'confidence': obj.get('confidence', None)
            })
    
    return pd.DataFrame(predictions)


def load_gold_labels(gold_file: str) -> pd.DataFrame:
    """
    Load gold labels from JSONL file.
    
    Expected format per line:
    {
        "id": "unique_identifier",
        "sentence": "text content",
        "label": 0 or 1,
        "language": "en/es/fr/etc",
        "content_effect": 0 or 1 (optional),
        "metadata": {...}
    }
    
    Args:
        gold_file: Path to gold labels JSONL file
        
    Returns:
        DataFrame with all fields
    """
    gold_data = []
    with jsonlines.open(gold_file) as reader:
        for obj in reader:
            gold_data.append(obj)
    
    return pd.DataFrame(gold_data)


def calculate_accuracy(predictions: pd.DataFrame, gold: pd.DataFrame) -> float:
    """
    Calculate overall accuracy.
    
    Accuracy = (TP + TN) / Total
    
    Args:
        predictions: DataFrame with 'id' and 'label' columns
        gold: DataFrame with 'id' and 'label' columns
        
    Returns:
        Accuracy score (0-1)
    """
    # Merge on id to align predictions with gold labels
    merged = pd.merge(predictions, gold, on='id', suffixes=('_pred', '_gold'))
    
    if len(merged) == 0:
        raise ValueError("No matching IDs between predictions and gold labels")
    
    correct = (merged['label_pred'] == merged['label_gold']).sum()
    total = len(merged)
    
    accuracy = correct / total
    return accuracy


def calculate_tce(predictions: pd.DataFrame, gold: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate True Content Effect (TCE) metric.
    
    TCE measures the model's ability to detect plausibility changes
    caused by content manipulation while controlling for surface features.
    
    TCE = Accuracy on instances where content_effect = 1
    
    Args:
        predictions: DataFrame with 'id' and 'label' columns
        gold: DataFrame with 'id', 'label', and 'content_effect' columns
        
    Returns:
        Dictionary with TCE score and related metrics
    """
    merged = pd.merge(predictions, gold, on='id', suffixes=('_pred', '_gold'))
    
    # Check if content_effect column exists
    if 'content_effect' not in merged.columns:
        print("Warning: 'content_effect' column not found in gold labels. Skipping TCE calculation.")
        return {
            'tce': None,
            'tce_count': 0,
            'tce_accuracy': None
        }
    
    # Filter instances with content effect
    tce_instances = merged[merged['content_effect'] == 1]
    
    if len(tce_instances) == 0:
        return {
            'tce': None,
            'tce_count': 0,
            'tce_accuracy': None
        }
    
    # Calculate accuracy on TCE instances
    tce_correct = (tce_instances['label_pred'] == tce_instances['label_gold']).sum()
    tce_total = len(tce_instances)
    tce_accuracy = tce_correct / tce_total
    
    return {
        'tce': tce_accuracy,
        'tce_count': tce_total,
        'tce_accuracy': tce_accuracy
    }


def calculate_intra_plausibility(predictions: pd.DataFrame, gold: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate intra-lingual plausibility consistency.
    
    Measures consistency of plausibility judgments within the same language.
    For similar sentences in the same language, predictions should be consistent.
    
    Args:
        predictions: DataFrame with predictions
        gold: DataFrame with gold labels and language info
        
    Returns:
        Dictionary with per-language consistency scores
    """
    merged = pd.merge(predictions, gold, on='id', suffixes=('_pred', '_gold'))
    
    if 'language' not in merged.columns:
        print("Warning: 'language' column not found. Skipping intra-lingual analysis.")
        return {}
    
    # Calculate accuracy per language
    language_scores = {}
    for lang in merged['language'].unique():
        lang_data = merged[merged['language'] == lang]
        correct = (lang_data['label_pred'] == lang_data['label_gold']).sum()
        total = len(lang_data)
        language_scores[lang] = correct / total if total > 0 else 0.0
    
    # Overall intra-lingual score (average across languages)
    if language_scores:
        language_scores['average'] = np.mean(list(language_scores.values()))
    
    return language_scores


def calculate_cross_plausibility(predictions: pd.DataFrame, gold: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate cross-lingual plausibility consistency.
    
    Measures consistency of plausibility judgments across languages.
    For translations of the same content, predictions should be consistent.
    
    This assumes gold data contains a 'content_id' field that links
    translations of the same content across languages.
    
    Args:
        predictions: DataFrame with predictions
        gold: DataFrame with gold labels, language, and content_id
        
    Returns:
        Dictionary with cross-lingual consistency metrics
    """
    merged = pd.merge(predictions, gold, on='id', suffixes=('_pred', '_gold'))
    
    if 'content_id' not in merged.columns:
        print("Warning: 'content_id' column not found. Skipping cross-lingual analysis.")
        return {'cross_lingual_consistency': None}
    
    # Group by content_id to find translations
    grouped = merged.groupby('content_id')
    
    consistent_count = 0
    total_content_groups = 0
    
    for content_id, group in grouped:
        if len(group) > 1:  # Must have at least 2 translations
            total_content_groups += 1
            # Check if all predictions are the same
            if group['label_pred'].nunique() == 1:
                consistent_count += 1
    
    consistency = consistent_count / total_content_groups if total_content_groups > 0 else None
    
    return {
        'cross_lingual_consistency': consistency,
        'content_groups_evaluated': total_content_groups,
        'consistent_groups': consistent_count
    }


def calculate_confusion_matrix(predictions: pd.DataFrame, gold: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate confusion matrix metrics.
    
    Returns:
        Dictionary with TP, TN, FP, FN counts and derived metrics
    """
    merged = pd.merge(predictions, gold, on='id', suffixes=('_pred', '_gold'))
    
    # Calculate confusion matrix components
    tp = ((merged['label_pred'] == 1) & (merged['label_gold'] == 1)).sum()
    tn = ((merged['label_pred'] == 0) & (merged['label_gold'] == 0)).sum()
    fp = ((merged['label_pred'] == 1) & (merged['label_gold'] == 0)).sum()
    fn = ((merged['label_pred'] == 0) & (merged['label_gold'] == 1)).sum()
    
    # Calculate derived metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def run_evaluation(pred_file: str, gold_file: str, output_dir: Optional[str] = None) -> Dict:
    """
    Main evaluation pipeline.
    
    Runs all evaluation metrics and optionally saves results to file.
    
    Args:
        pred_file: Path to predictions JSONL file
        gold_file: Path to gold labels JSONL file
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    print(f"Loading predictions from: {pred_file}")
    predictions = load_predictions(pred_file)
    print(f"Loaded {len(predictions)} predictions")
    
    print(f"Loading gold labels from: {gold_file}")
    gold = load_gold_labels(gold_file)
    print(f"Loaded {len(gold)} gold labels")
    
    # Run all metrics
    results = {}
    
    # 1. Accuracy
    print("\nCalculating accuracy...")
    results['accuracy'] = calculate_accuracy(predictions, gold)
    print(f"  Accuracy: {results['accuracy']:.4f}")
    
    # 2. Confusion Matrix
    print("\nCalculating confusion matrix...")
    cm_results = calculate_confusion_matrix(predictions, gold)
    results.update(cm_results)
    print(f"  Precision: {cm_results['precision']:.4f}")
    print(f"  Recall: {cm_results['recall']:.4f}")
    print(f"  F1 Score: {cm_results['f1_score']:.4f}")
    
    # 3. True Content Effect (TCE)
    print("\nCalculating True Content Effect (TCE)...")
    tce_results = calculate_tce(predictions, gold)
    results['tce_metrics'] = tce_results
    if tce_results['tce'] is not None:
        print(f"  TCE: {tce_results['tce']:.4f} (n={tce_results['tce_count']})")
    
    # 4. Intra-lingual Plausibility
    print("\nCalculating intra-lingual plausibility...")
    intra_results = calculate_intra_plausibility(predictions, gold)
    results['intra_lingual'] = intra_results
    if intra_results:
        print(f"  Average intra-lingual consistency: {intra_results.get('average', 'N/A'):.4f}")
        for lang, score in intra_results.items():
            if lang != 'average':
                print(f"    {lang}: {score:.4f}")
    
    # 5. Cross-lingual Plausibility
    print("\nCalculating cross-lingual plausibility...")
    cross_results = calculate_cross_plausibility(predictions, gold)
    results['cross_lingual'] = cross_results
    if cross_results['cross_lingual_consistency'] is not None:
        print(f"  Cross-lingual consistency: {cross_results['cross_lingual_consistency']:.4f}")
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "metrics.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for SemEval 2026 Task 11"
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSONL file'
    )
    parser.add_argument(
        '--gold',
        type=str,
        required=True,
        help='Path to gold labels JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (optional)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_evaluation(args.predictions, args.gold, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    if results['tce_metrics']['tce'] is not None:
        print(f"TCE: {results['tce_metrics']['tce']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()