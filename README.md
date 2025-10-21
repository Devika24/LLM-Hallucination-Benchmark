# LLM Hallucination Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/)

A rigorous evaluation framework for measuring factual accuracy and hallucination in Large Language Models (LLMs) using Wikipedia and ArXiv references with semantic similarity scoring.

## 🎯 Overview

Large Language Models often generate responses that deviate from factual information or training data, a phenomenon known as "hallucination." This benchmark provides a comprehensive framework to:

- **Detect Multi-dimensional Hallucinations**: Identify factual inconsistencies, contradictions, and unverifiable claims
- **Quantify Semantic Similarity**: Use Sentence-BERT embeddings to measure alignment with ground truth
- **Evaluate Factual Consistency**: Score model outputs against verified reference documents
- **Visualize Performance**: Interactive dashboards for comparative analysis across models

### Key Features

- 🔍 **Multi-modal Detection**: Supports both intrinsic and extrinsic hallucination evaluation
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, BLEU, ROUGE, and BERTScore
- 🎨 **Visualization Tools**: Confusion matrices, ROC curves, and comparative bar charts
- 🔄 **Dynamic Test Generation**: Prevents data leakage and ensures robust evaluation
- 📈 **Batch Processing**: Efficient evaluation of multiple models simultaneously

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Dataset Structure](#dataset-structure)
- [Usage Examples](#usage-examples)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
