"""
Comprehensive Metrics Calculator for Hallucination Evaluation

Implements various metrics for evaluating hallucination detection including:
- Classification metrics (Accuracy, Precision, Recall, F1)
- Text similarity metrics (BLEU, ROUGE, BERTScore)
- Calibration metrics (ECE, Brier Score)

Author: Your Name
Date: 2024
License: MIT
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for hallucination evaluation.
    
    This class provides methods to calculate various evaluation metrics
    used in NLP and hallucination detection tasks.
    """
    
    def __init__(self):
        """Initialize metrics calculator with necessary components."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()
        logger.info("Metrics calculator initialized")
    
    def classification_metrics(
        self,
        y_true: List[bool],
        y_pred: List[bool]
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Computes standard binary classification metrics:
        - Accuracy: Overall correctness
        - Precision: Correct positive predictions / Total positive predictions
        - Recall: Correct positive predictions / Total actual positives
        - F1 Score: Harmonic mean of precision and recall
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with all classification metrics
        """
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            })
        
        return metrics
    
    def bleu_score(
        self,
        predictions: List[str],
        references: List[str],
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score for text similarity.
        
        BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
        between predicted and reference texts.
        
        Formula:
        BLEU = BP × exp(Σ wₙ log pₙ)
        where pₙ is n-gram precision and BP is brevity penalty
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            max_n: Maximum n-gram size (default: 4)
            
        Returns:
            Average BLEU score across all samples
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            
            # Calculate BLEU with smoothing
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                smoothing_function=self.smoothing.method1
            )
            scores.append(score)
        
        return float(np.mean(scores))
    
    def rouge_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores for text similarity.
        
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
        recall-based n-gram overlap.
        
        Variants:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': float(np.mean(rouge1_scores)),
            'rouge2': float(np.mean(rouge2_scores)),
            'rougeL': float(np.mean(rougeL_scores))
        }
    
    def bert_score(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = 'en'
    ) -> float:
        """
        Calculate BERTScore for semantic similarity.
        
        BERTScore uses contextual embeddings from BERT to compute
        token-level similarity between predictions and references.
        
        Formula:
        BERTScore = (1/|x|) Σ max cos_sim(xᵢ, yⱼ)
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            lang: Language code (default: 'en')
            
        Returns:
            Average BERTScore F1 across all samples
        """
        try:
            P, R, F1 = bert_score_calc(
                predictions,
                references,
                lang=lang,
                verbose=False
            )
            return float(F1.mean().item())
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return 0.0
    
    def expected_calibration_error(
        self,
        y_true: List[bool],
        y_probs: List[float],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures how well predicted probabilities match actual outcomes.
        Lower values indicate better calibration.
        
        Formula:
        ECE = Σ (|Bₘ|/n) |acc(Bₘ) - conf(Bₘ)|
        
        Args:
            y_true: Ground truth labels
            y_probs: Predicted probabilities
            n_bins: Number of calibration bins
            
        Returns:
            Expected Calibration Error
        """
        y_true = np.array(y_true, dtype=int)
        y_probs = np.array(y_probs)
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_probs, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_probs[mask].mean()
                bin_size = mask.sum()
                ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)
        
        return float(ece)
    
    def brier_score(
        self,
        y_true: List[bool],
        y_probs: List[float]
    ) -> float:
        """
        Calculate Brier Score for probabilistic predictions.
        
        Brier Score measures the accuracy of probabilistic predictions.
        Lower values indicate better calibration.
        
        Formula:
        BS = (1/N) Σ (fᵢ - oᵢ)²
        where fᵢ is predicted probability and oᵢ is actual outcome
        
        Args:
            y_true: Ground truth labels
            y_probs: Predicted probabilities
            
        Returns:
            Brier Score
        """
        y_true = np.array(y_true, dtype=float)
        y_probs = np.array(y_probs)
        
        return float(np.mean((y_probs - y_true) ** 2))
    
    def roc_metrics(
        self,
        y_true: List[bool],
        y_scores: List[float]
    ) -> Dict[str, any]:
        """
        Calculate ROC curve metrics.
        
        Computes:
        - ROC-AUC: Area under the ROC curve
        - FPR: False positive rates at various thresholds
        - TPR: True positive rates at various thresholds
        - Thresholds: Decision thresholds
        
        Args:
            y_true: Ground truth labels
            y_scores: Predicted scores/probabilities
            
        Returns:
            Dictionary with ROC metrics and curve data
        """
        y_true = np.array(y_true, dtype=int)
        y_scores = np.array(y_scores)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calculate AUC
        auc = roc_auc_score(y_true, y_scores)
        
        return {
            'roc_auc': float(auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def hallucination_specific_metrics(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate hallucination-specific metrics.
        
        Includes:
        - Intrinsic hallucination rate
        - Extrinsic hallucination rate
        - Factual accuracy
        - Consistency score
        
        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            
        Returns:
            Dictionary with hallucination-specific metrics
        """
        intrinsic_halluc = []
        extrinsic_halluc = []
        factual_correct = []
        
        for pred, gt in zip(predictions, ground_truths):
            # Intrinsic: contradicts input context
            if 'intrinsic_hallucination' in gt:
                intrinsic_halluc.append(
                    pred.get('has_intrinsic_hallucination', False) == 
                    gt['intrinsic_hallucination']
                )
            
            # Extrinsic: contradicts world knowledge
            if 'extrinsic_hallucination' in gt:
                extrinsic_halluc.append(
                    pred.get('has_extrinsic_hallucination', False) == 
                    gt['extrinsic_hallucination']
                )
            
            # Factual correctness
            if 'factual_accuracy' in gt:
                factual_correct.append(
                    pred.get('factual_score', 0) >= gt['factual_accuracy']
                )
        
        metrics = {}
        
        if intrinsic_halluc:
            metrics['intrinsic_detection_accuracy'] = float(np.mean(intrinsic_halluc))
        
        if extrinsic_halluc:
            metrics['extrinsic_detection_accuracy'] = float(np.mean(extrinsic_halluc))
        
        if factual_correct:
            metrics['factual_accuracy'] = float(np.mean(factual_correct))
        
        return metrics
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate metrics across multiple runs or models.
        
        Calculates mean, std, min, and max for each metric.
        
        Args:
            all_metrics: List of metric dictionaries
            
        Returns:
            Aggregated statistics for each metric
        """
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in all_metrics:
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values and isinstance(values[0], (int, float)):
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return aggregated