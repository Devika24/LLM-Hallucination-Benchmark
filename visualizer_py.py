"""
Visualization Module for Hallucination Benchmark Results

Creates publication-quality visualizations including:
- Confusion matrices
- ROC and PR curves
- Performance comparison charts
- Distribution plots

Author: Your Name
Date: 2024
License: MIT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import logging

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Comprehensive visualization toolkit for hallucination evaluation results.
    
    Generates various plots and charts for analysis and presentation.
    """
    
    def __init__(self, output_dir: str = "results/visualizations/"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Visualizer initialized. Outputs will be saved to: {output_dir}")
    
    def plot_confusion_matrix(
        self,
        y_true: List[bool],
        y_pred: List[bool],
        filename: str = "confusion_matrix.