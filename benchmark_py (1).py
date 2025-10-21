"""
LLM Hallucination Benchmark - Main Evaluation Script

This module provides the core functionality for evaluating Large Language Models
for hallucination detection using multiple metrics and datasets.

Author: Your Name
Date: 2024
License: MIT
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# Import custom modules
from data_loader import load_dataset
from llm_interface import LLMInterface
from hallucination_detector import HallucinationDetector
from metrics import MetricsCalculator
from visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HallucinationBenchmark:
    """
    Main benchmark class for evaluating LLM hallucinations.
    
    This class orchestrates the entire evaluation pipeline including:
    - Data loading and preprocessing
    - LLM response generation
    - Hallucination detection
    - Metrics calculation
    - Results visualization
    
    Attributes:
        model_name (str): Name of the LLM to evaluate
        dataset_name (str): Name of the benchmark dataset
        similarity_threshold (float): Threshold for semantic similarity
        batch_size (int): Number of samples to process in parallel
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        dataset_name: str = "truthfulqa",
        similarity_threshold: float = 0.75,
        batch_size: int = 32,
        output_dir: str = "results/",
        api_key: Optional[str] = None
    ):
        """
        Initialize the benchmark with specified parameters.
        
        Args:
            model_name: LLM model identifier
            dataset_name: Benchmark dataset name
            similarity_threshold: Minimum similarity score for non-hallucination
            batch_size: Batch size for processing
            output_dir: Directory to save results
            api_key: API key for LLM provider (optional, can use env var)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        logger.info(f"Initializing benchmark for model: {model_name}")
        self.llm = LLMInterface(model_name, api_key)
        self.detector = HallucinationDetector(threshold=similarity_threshold)
        self.metrics = MetricsCalculator()
        self.visualizer = Visualizer(output_dir)
        
        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)
        
        # Results storage
        self.results = {
            'model': model_name,
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'predictions': [],
            'metrics': {}
        }
    
    def evaluate(self, num_samples: Optional[int] = None) -> Dict:
        """
        Run the complete evaluation pipeline.
        
        This method:
        1. Generates LLM responses for each query
        2. Detects hallucinations using semantic similarity
        3. Calculates comprehensive evaluation metrics
        4. Generates visualizations
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary containing all evaluation results and metrics
        """
        logger.info("Starting evaluation...")
        
        # Limit samples if specified
        dataset = self.dataset[:num_samples] if num_samples else self.dataset
        
        # Initialize tracking variables
        predictions = []
        ground_truths = []
        responses = []
        references = []
        
        # Process each sample
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Extract query and reference
                query = sample['query']
                reference = sample['reference']
                ground_truth = sample.get('is_hallucination', False)
                
                # Generate LLM response
                response = self.llm.generate(query)
                
                # Detect hallucination
                detection_result = self.detector.detect(response, reference)
                
                # Store results
                predictions.append(detection_result['is_hallucination'])
                ground_truths.append(ground_truth)
                responses.append(response)
                references.append(reference)
                
                # Store detailed results
                self.results['predictions'].append({
                    'id': sample.get('id', idx),
                    'query': query,
                    'response': response,
                    'reference': reference,
                    'predicted_hallucination': detection_result['is_hallucination'],
                    'actual_hallucination': ground_truth,
                    'similarity_score': detection_result['similarity'],
                    'factual_errors': detection_result.get('errors', [])
                })
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        self.results['metrics'] = self._calculate_all_metrics(
            predictions,
            ground_truths,
            responses,
            references
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        self._generate_visualizations()
        
        # Save results
        self._save_results()
        
        logger.info("Evaluation complete!")
        return self.results
    
    def _calculate_all_metrics(
        self,
        predictions: List[bool],
        ground_truths: List[bool],
        responses: List[str],
        references: List[str]
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Metrics include:
        - Classification metrics (Accuracy, Precision, Recall, F1)
        - Text similarity metrics (BLEU, ROUGE, BERTScore)
        - Calibration metrics (ECE, Brier Score)
        - Aggregate statistics
        
        Args:
            predictions: Binary hallucination predictions
            ground_truths: Actual hallucination labels
            responses: LLM generated responses
            references: Ground truth references
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Classification metrics
        classification = self.metrics.classification_metrics(
            ground_truths,
            predictions
        )
        metrics.update(classification)
        
        # Hallucination rate
        metrics['hallucination_rate'] = np.mean(predictions)
        
        # Text similarity metrics (for non-hallucinated samples)
        non_halluc_indices = [i for i, p in enumerate(predictions) if not p]
        if non_halluc_indices:
            non_halluc_responses = [responses[i] for i in non_halluc_indices]
            non_halluc_references = [references[i] for i in non_halluc_indices]
            
            # BLEU score
            metrics['bleu'] = self.metrics.bleu_score(
                non_halluc_responses,
                non_halluc_references
            )
            
            # ROUGE scores
            rouge = self.metrics.rouge_scores(
                non_halluc_responses,
                non_halluc_references
            )
            metrics.update(rouge)
            
            # BERTScore
            metrics['bertscore'] = self.metrics.bert_score(
                non_halluc_responses,
                non_halluc_references
            )
        
        # Semantic similarity distribution
        similarities = [
            self.detector.calculate_similarity(r, ref)
            for r, ref in zip(responses, references)
        ]
        metrics['avg_similarity'] = np.mean(similarities)
        metrics['std_similarity'] = np.std(similarities)
        metrics['min_similarity'] = np.min(similarities)
        metrics['max_similarity'] = np.max(similarities)
        
        return metrics
    
    def _generate_visualizations(self):
        """Generate all visualization outputs."""
        predictions = [p['predicted_hallucination'] for p in self.results['predictions']]
        ground_truths = [p['actual_hallucination'] for p in self.results['predictions']]
        
        # Confusion matrix
        self.visualizer.plot_confusion_matrix(
            ground_truths,
            predictions,
            filename='confusion_matrix.png'
        )
        
        # ROC curve
        similarities = [p['similarity_score'] for p in self.results['predictions']]
        self.visualizer.plot_roc_curve(
            ground_truths,
            similarities,
            filename='roc_curve.png'
        )
        
        # Metrics comparison
        self.visualizer.plot_metrics_comparison(
            self.results['metrics'],
            filename='metrics_comparison.png'
        )
        
        # Similarity distribution
        self.visualizer.plot_similarity_distribution(
            similarities,
            predictions,
            filename='similarity_distribution.png'
        )
    
    def _save_results(self):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{self.dataset_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        
        # Also save a summary
        summary_path = os.path.join(self.output_dir, f"summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"LLM Hallucination Benchmark Results\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"{'-'*50}\n")
            for key, value in self.results['metrics'].items():
                if isinstance(value, float):
                    f.write(f"{key:30s}: {value:.4f}\n")
                else:
                    f.write(f"{key:30s}: {value}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LLM Hallucination Benchmark Evaluation'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='LLM model name (default: gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='truthfulqa',
        help='Dataset name (default: truthfulqa)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.75,
        help='Similarity threshold (default: 0.75)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/',
        help='Output directory (default: results/)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key for LLM provider'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize benchmark
    benchmark = HallucinationBenchmark(
        model_name=args.model,
        dataset_name=args.dataset,
        similarity_threshold=args.threshold,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        api_key=args.api_key
    )
    
    # Run evaluation
    results = benchmark.evaluate(num_samples=args.num_samples)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"\nKey Metrics:")
    print(f"  Hallucination Rate: {results['metrics']['hallucination_rate']:.2%}")
    print(f"  Accuracy:           {results['metrics']['accuracy']:.4f}")
    print(f"  Precision:          {results['metrics']['precision']:.4f}")
    print(f"  Recall:             {results['metrics']['recall']:.4f}")
    print(f"  F1 Score:           {results['metrics']['f1_score']:.4f}")
    if 'bertscore' in results['metrics']:
        print(f"  BERTScore:          {results['metrics']['bertscore']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()