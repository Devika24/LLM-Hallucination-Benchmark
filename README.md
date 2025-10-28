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
- 🎨 **Visualization Techniques**: Confusion matrices, ROC curves, and comparative bar charts
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
- CUDA-compatible GPU (recommended for faster inference)
- 8GB+ RAM

### Step 1: Clone the Repository

```bash
git clone https://github.com/Devika24/LLM-Hallucination-Benchmark.git
cd LLM-Hallucination-Benchmark
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Required Models

```bash
python scripts/download_models.py
```

This will download:
- Sentence-BERT model (`all-MiniLM-L6-v2`)
- BERTScore model
- Reference datasets (TruthfulQA, FEVER, etc.)

---

## ⚡ Quick Start

### Basic Evaluation

```bash
python src/benchmark.py --model gpt-3.5-turbo --dataset truthfulqa
```

### Advanced Configuration

```bash
python src/benchmark.py \
    --model gpt-4 \
    --dataset fever \
    --batch_size 32 \
    --output_dir results/ \
    --visualize
```

### Evaluate Multiple Models

```bash
python src/batch_evaluate.py \
    --models gpt-3.5-turbo gpt-4 claude-2 \
    --datasets truthfulqa fever halueval \
    --save_results
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Prompt                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Response Generation                    │
│  (GPT-3.5, GPT-4, Claude, LLaMA, etc.)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│              Hallucination Detection Pipeline              │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Semantic  │  │   Factual    │  │ Consistency  │        │
│  │ Similarity │  │   Accuracy   │  │   Scoring    │        │
│  └────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Evaluation Metrics                         │
│  • Hallucination Rate  • F1 Score  • BERTScore              │
│  • Precision/Recall    • BLEU      • ROUGE                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Visualization & Analysis Dashboard               │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Data Loader** (`src/data_loader.py`)
- Loads benchmark datasets (TruthfulQA, FEVER, HaluEval)
- Preprocesses queries and reference documents
- Handles Wikipedia and ArXiv source integration
- Implements data augmentation for robustness testing

#### 2. **LLM Interface** (`src/llm_interface.py`)
- Unified API for multiple LLM providers
- Supports OpenAI, Anthropic, Hugging Face models
- Token counting and cost estimation
- Retry logic with exponential backoff

#### 3. **Hallucination Detector** (`src/hallucination_detector.py`)

**Semantic Similarity Module**
- Uses Sentence-BERT for embedding generation
- Cosine similarity computation
- Threshold-based classification
- Handles multi-sentence contexts

```python
# Semantic similarity calculation
similarity_score = cosine_similarity(
    model.encode(llm_response),
    model.encode(reference_text)
)
is_hallucination = similarity_score < threshold  # Default: 0.75
```

**Factual Consistency Checker**
- Entity extraction and verification
- Claim decomposition
- Evidence retrieval from knowledge bases
- Contradiction detection

**Intrinsic vs Extrinsic Detection**
- **Intrinsic**: Checks consistency with input context
- **Extrinsic**: Verifies against external knowledge (Wikipedia/ArXiv)

#### 4. **Metrics Calculator** (`src/metrics.py`)
- Binary classification metrics (Precision, Recall, F1)
- Text similarity metrics (BLEU, ROUGE)
- Semantic metrics (BERTScore)
- Calibration metrics (ECE, Brier Score)

#### 5. **Visualizer** (`src/visualizer.py`)
- Confusion matrices with seaborn
- ROC/PR curves
- Model comparison charts
- Interactive HTML dashboards

---

## 📊 Dataset Structure

### Supported Datasets

| Dataset | Size | Task Type | Hallucination Type |
|---------|------|-----------|-------------------|
| TruthfulQA | 817 | QA | Extrinsic |
| FEVER | 145K | Fact Verification | Both |
| HaluEval | 35K | Multi-task | Both |
| Custom | Variable | User-defined | Configurable |

### Data Format

```json
{
  "id": "unique_identifier",
  "query": "Question or prompt text",
  "reference": "Ground truth or source document",
  "category": "science|history|politics",
  "difficulty": "easy|medium|hard",
  "metadata": {
    "source": "wikipedia",
    "url": "https://...",
    "timestamp": "2024-01-15"
  }
}
```

### Adding Custom Datasets

```python
from src.data_loader import CustomDataset

dataset = CustomDataset(
    path='data/custom_dataset.json',
    format='json',
    preprocessing=True
)
```

---

## 💻 Usage Examples

### Example 1: Single Model Evaluation

```python
from src.benchmark import HallucinationBenchmark

# Initialize benchmark
benchmark = HallucinationBenchmark(
    model_name='gpt-3.5-turbo',
    dataset='truthfulqa',
    similarity_threshold=0.75
)

# Run evaluation
results = benchmark.evaluate()

# Display results
print(f"Hallucination Rate: {results['hallucination_rate']:.2%}")
print(f"F1 Score: {results['f1_score']:.3f}")
print(f"BERTScore: {results['bertscore']:.3f}")
```

### Example 2: Comparative Analysis

```python
from src.batch_evaluate import compare_models

models = ['gpt-3.5-turbo', 'gpt-4', 'claude-2']
results = compare_models(
    models=models,
    dataset='fever',
    metrics=['accuracy', 'f1', 'bertscore']
)

# Generate comparison report
results.plot_comparison(save_path='results/comparison.png')
```

### Example 3: Custom Hallucination Detection

```python
from src.hallucination_detector import HallucinationDetector

detector = HallucinationDetector(model='sentence-transformers/all-MiniLM-L6-v2')

response = "The Eiffel Tower was built in 1887 and is 330 meters tall."
reference = "The Eiffel Tower was built in 1889 and is 324 meters tall."

result = detector.detect(response, reference)
print(f"Hallucination Detected: {result['is_hallucination']}")
print(f"Similarity Score: {result['similarity']:.3f}")
print(f"Factual Errors: {result['errors']}")
```

---

## 📈 Evaluation Metrics

### 1. **Hallucination Rate**
Percentage of responses containing factual inconsistencies:
```
Hallucination Rate = (False Claims / Total Claims) × 100%
```

### 2. **Semantic Similarity Score**
Cosine similarity between response and reference embeddings:
```
Similarity = cos(θ) = (A · B) / (||A|| ||B||)
```

### 3. **F1 Score**
Harmonic mean of precision and recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 4. **BERTScore**
Token-level semantic similarity using contextual embeddings:
```
BERTScore = Σ max cos_sim(token_ref, token_hyp)
```

### 5. **Calibration Metrics**
- **Expected Calibration Error (ECE)**: Measures confidence calibration
- **Brier Score**: Assesses probabilistic predictions

---

## 🎨 Results

### Benchmark Results (Sample)

| Model | Hallucination Rate | F1 Score | BERTScore | Avg. Latency |
|-------|-------------------|----------|-----------|--------------|
| GPT-4 | 8.2% | 0.894 | 0.912 | 1.2s |
| GPT-3.5-Turbo | 15.7% | 0.831 | 0.876 | 0.8s |
| Claude-2 | 10.3% | 0.867 | 0.891 | 1.1s |
| LLaMA-2-70B | 18.9% | 0.798 | 0.845 | 2.3s |

### Visualization Examples

The benchmark generates:
- **Confusion Matrices**: True/False positive rates
- **ROC Curves**: Trade-offs between sensitivity and specificity
- **Model Comparison Charts**: Side-by-side performance metrics
- **Error Analysis Plots**: Common hallucination patterns

---

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{llm_hallucination_benchmark_2024,
  author = {Your Name},
  title = {LLM Hallucination Benchmark: A Comprehensive Evaluation Framework},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Devika24/LLM-Hallucination-Benchmark}}
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- TruthfulQA dataset creators
- Sentence-Transformers library
- OpenAI, Anthropic, and Hugging Face for model APIs
- Research community for hallucination detection methods

---

## 📧 Contact

- **Author**: [Devika A]
- **Email**: devika8195@gmail.com
- **GitHub**: [@Devika24](https://github.com/Devika24)

---

## 🗺️ Roadmap

- [ ] Add multilingual support
- [ ] Implement real-time monitoring dashboard
- [ ] Support for multi-modal hallucination (vision-language)
- [ ] Integration with more LLM providers
- [ ] Automated report generation
