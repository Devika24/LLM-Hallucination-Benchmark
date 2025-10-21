# Complete Setup Guide for LLM Hallucination Benchmark

This guide provides step-by-step instructions to set up and run the LLM Hallucination Benchmark.

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM (16GB recommended)
- Optional: CUDA-compatible GPU for faster processing

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/Devika24/LLM-Hallucination-Benchmark.git
cd LLM-Hallucination-Benchmark
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Step 4: Download Required Models

```bash
# Download Sentence-BERT model
python scripts/download_models.py

# Or manually with Python
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Step 5: Set Up API Keys

Create a `.env` file in the project root:

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# Anthropic API Key
ANTHROPIC_API_KEY=your-anthropic-key-here

# Hugging Face Token (optional)
HUGGINGFACE_TOKEN=your-hf-token-here
```

### Step 6: Download Datasets

```bash
# Download all benchmark datasets
python scripts/download_datasets.py

# Or download individually
python scripts/download_datasets.py --dataset truthfulqa
python scripts/download_datasets.py --dataset fever
python scripts/download_datasets.py --dataset halueval
```

## 🚀 Quick Start

### Basic Evaluation

```bash
# Evaluate GPT-3.5-Turbo on TruthfulQA
python src/benchmark.py \
    --model gpt-3.5-turbo \
    --dataset truthfulqa \
    --output_dir results/

# View results
cat results/summary_*.txt
```

### Advanced Usage

```bash
# Custom configuration
python src/benchmark.py \
    --model gpt-4 \
    --dataset fever \
    --threshold 0.80 \
    --batch_size 16 \
    --num_samples 100 \
    --visualize

# Multiple models comparison
python src/batch_evaluate.py \
    --models gpt-3.5-turbo gpt-4 claude-2 \
    --datasets truthfulqa fever \
    --save_results
```

## 📁 Project Structure

```
LLM-Hallucination-Benchmark/
├── src/
│   ├── benchmark.py              # Main evaluation script
│   ├── hallucination_detector.py # Detection methods
│   ├── llm_interface.py          # LLM API wrapper
│   ├── data_loader.py            # Dataset loaders
│   ├── metrics.py                # Evaluation metrics
│   └── visualizer.py             # Visualization tools
├── data/
│   ├── truthfulqa/               # TruthfulQA dataset
│   ├── fever/                    # FEVER dataset
│   ├── halueval/                 # HaluEval dataset
│   └── custom/                   # Custom datasets
├── results/
│   ├── visualizations/           # Generated plots
│   └── *.json                    # Results files
├── tests/
│   ├── test_detector.py
│   ├── test_metrics.py
│   └── test_benchmark.py
├── scripts/
│   ├── download_models.py
│   ├── download_datasets.py
│   └── run_experiments.sh
├── config/
│   └── config.yaml               # Configuration file
├── docs/
│   ├── API.md                    # API documentation
│   ├── PAPER_TEMPLATE.md         # Research paper template
│   └── EXAMPLES.md               # Usage examples
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── .env.example
```

## 🔑 Configuration

### Using config.yaml

Edit `config/config.yaml` to customize:

```yaml
models:
  gpt-4:
    temperature: 0.0
    max_tokens: 4096

detection:
  semantic:
    threshold: 0.75
    model: "sentence-transformers/all-MiniLM-L6-v2"

output:
  base_dir: "results/"
  save_predictions: true
```

### Command Line Override

```bash
python src/benchmark.py \
    --config config/custom_config.yaml \
    --threshold 0.80 \
    --output_dir custom_results/
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 📊 Understanding Results

### Output Files

After running evaluation, you'll find:

```
results/
├── gpt-3.5-turbo_truthfulqa_20241021_143000.json
├── summary_20241021_143000.txt
└── visualizations/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── metrics_comparison.png
    ├── similarity_distribution.png
    └── dashboard.html
```

### Interpreting Metrics

**Hallucination Rate:** Lower is better (target: <10%)
- Percentage of responses containing factual errors

**F1 Score:** Higher is better (target: >0.85)
- Balance between precision and recall

**BERTScore:** Higher is better (target: >0.88)
- Semantic similarity to reference

**Accuracy:** Higher is better (target: >0.90)
- Overall correctness of hallucination detection

## 🐛 Troubleshooting

### Common Issues

**Issue: API Rate Limit Exceeded**
```bash
# Solution: Add delay between requests
python src/benchmark.py --rate_limit 20  # 20 requests per minute
```

**Issue: Out of Memory**
```bash
# Solution: Reduce batch size
python src/benchmark.py --batch_size 8
```

**Issue: CUDA Out of Memory**
```bash
# Solution: Use CPU or smaller model
python src/benchmark.py --device cpu
```

**Issue: Model Download Fails**
```bash
# Solution: Manual download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('models/')"
```

### Getting Help

1. Check [Issues](https://github.com/Devika24/LLM-Hallucination-Benchmark/issues)
2. Read [Documentation](https://github.com/Devika24/LLM-Hallucination-Benchmark/docs)
3. Join [Discussions](https://github.com/Devika24/LLM-Hallucination-Benchmark/discussions)
4. Email maintainers (see README)

## 🔄 Updating the Benchmark

### Pull Latest Changes

```bash
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Updating Datasets

```bash
# Re-download datasets
python scripts/download_datasets.py --force-update
```

## 📈 Performance Optimization

### Speed Up Evaluation

1. **Use GPU:** Set `--device cuda`
2. **Increase Batch Size:** `--batch_size 64` (if memory allows)
3. **Enable Caching:** Embeddings are cached automatically
4. **Parallel Processing:** Set in config.yaml

### Reduce Costs

1. **Use Smaller Models:** GPT-3.5-Turbo instead of GPT-4
2. **Limit Samples:** `--num_samples 100` for testing
3. **Cache Responses:** Set `--cache_responses true`

## 🎓 Example Workflows

### Research Workflow

```bash
# 1. Run comprehensive evaluation
python src/benchmark.py \
    --model gpt-4 \
    --dataset truthfulqa \
    --visualize

# 2. Generate comparison
python src/batch_evaluate.py \
    --models gpt-3.5-turbo gpt-4 claude-2 \
    --datasets truthfulqa

# 3. Create paper figures
python scripts/generate_paper_figures.py \
    --results_dir results/ \
    --output_dir paper_figures/

# 4. Export to LaTeX tables
python scripts/export_latex_tables.py \
    --results_dir results/
```

### Production Workflow

```bash
# 1. Test on sample data
python src/benchmark.py \
    --model your-model \
    --dataset custom \
    --num_samples 50

# 2. Full evaluation
python src/benchmark.py \
    --model your-model \
    --dataset custom \
    --config config/production.yaml

# 3. Monitor performance
python scripts/monitor.py \
    --results_dir results/ \
    --alert_threshold 0.15  # Alert if hallucination > 15%
```

## 🔐 Security Best Practices

### API Key Management

1. **Never commit `.env` file**
2. **Use environment variables in production**
3. **Rotate keys regularly**
4. **Use read-only keys when possible**

### Data Privacy

1. **Don't log sensitive data**
2. **Anonymize personal information**
3. **Use secure storage for results**

## 🌐 Deployment

### Docker Deployment

```bash
# Build image
docker build -t hallucination-benchmark .

# Run container
docker run -v $(pwd)/results:/app/results \
    --env-file .env \
    hallucination-benchmark \
    --model gpt-3.5-turbo \
    --dataset truthfulqa
```

### Cloud Deployment

**AWS Lambda:**
```bash
# Package for Lambda
pip install -r requirements.txt -t package/
cd package && zip -r ../deployment.zip .
cd .. && zip -g deployment.zip src/*.py

# Upload to Lambda
aws lambda create-function \
    --function-name hallucination-benchmark \
    --runtime python3.10 \
    --handler src.benchmark.lambda_handler \
    --zip-file fileb://deployment.zip
```

## 📖 Additional Resources

### Documentation

- [API Reference](docs/API.md)
- [Dataset Formats](docs/DATASETS.md)
- [Custom Models](docs/CUSTOM_MODELS.md)
- [Metrics Explained](docs/METRICS.md)

### Tutorials

- [Basic Tutorial](docs/tutorials/basic.md)
- [Advanced Techniques](docs/tutorials/advanced.md)
- [Custom Datasets](docs/tutorials/custom_datasets.md)
- [Model Fine-tuning](docs/tutorials/fine_tuning.md)

### Community

- GitHub Discussions
- Discord Server
- Monthly Office Hours
- Research Collaboration

## 📝 Citation

If you use this benchmark, please cite:

```bibtex
@software{llm_hallucination_benchmark_2024,
  author = {Your Name},
  title = {LLM Hallucination Benchmark},
  year = {2024},
  url = {https://github.com/Devika24/LLM-Hallucination-Benchmark}
}
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

**Questions?** Open an issue or start a discussion!

**Happy Benchmarking! 🚀**