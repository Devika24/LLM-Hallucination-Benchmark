# Contributing to LLM Hallucination Benchmark

Thank you for your interest in contributing to the LLM Hallucination Benchmark! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Bug Reports:** Submit detailed bug reports with reproducible examples
- **Feature Requests:** Propose new features or improvements
- **Code Contributions:** Submit pull requests with bug fixes or new features
- **Documentation:** Improve documentation, add examples, or fix typos
- **Dataset Contributions:** Add new benchmark datasets
- **Model Evaluations:** Run benchmarks on new models and share results

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/LLM-Hallucination-Benchmark.git
cd LLM-Hallucination-Benchmark

# Add upstream remote
git remote add upstream https://github.com/Devika24/LLM-Hallucination-Benchmark.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies including dev tools
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

## 📝 Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line Length:** Maximum 88 characters (Black default)
- **Imports:** Group into stdlib, third-party, local (use isort)
- **Docstrings:** Google style for all public functions/classes
- **Type Hints:** Required for all function signatures

### Example

```python
from typing import List, Dict, Optional
import numpy as np

def detect_hallucination(
    response: str,
    reference: str,
    threshold: float = 0.75
) -> Dict[str, any]:
    """
    Detect hallucinations in LLM response.
    
    Args:
        response: Generated text from LLM
        reference: Ground truth reference text
        threshold: Similarity threshold for detection
        
    Returns:
        Dictionary containing detection results with keys:
            - is_hallucination: Boolean detection result
            - similarity: Float similarity score
            - confidence: Float confidence in prediction
            
    Raises:
        ValueError: If response or reference is empty
        
    Example:
        >>> result = detect_hallucination("Paris is capital", "Paris is the capital of France")
        >>> result['is_hallucination']
        False
    """
    if not response or not reference:
        raise ValueError("Response and reference cannot be empty")
    
    # Implementation here
    pass
```

### Code Formatting

We use automated formatters:

```bash
# Format code with Black
black src/

# Sort imports with isort
isort src/

# Check code quality with flake8
flake8 src/

# Type checking with mypy
mypy src/
```

## 🧪 Testing Guidelines

### Writing Tests

- All new features must include tests
- Aim for >80% code coverage
- Use pytest for all tests
- Place tests in `tests/` directory

### Test Structure

```python
import pytest
from src.hallucination_detector import HallucinationDetector

class TestHallucinationDetector:
    """Test suite for HallucinationDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Fixture to create detector instance."""
        return HallucinationDetector(threshold=0.75)
    
    def test_detect_no_hallucination(self, detector):
        """Test detection when no hallucination present."""
        response = "Paris is the capital of France."
        reference = "Paris is France's capital city."
        
        result = detector.detect(response, reference)
        
        assert result['is_hallucination'] is False
        assert result['similarity'] > 0.75
    
    def test_detect_with_hallucination(self, detector):
        """Test detection when hallucination present."""
        response = "Paris is the capital of Germany."
        reference = "Paris is the capital of France."
        
        result = detector.detect(response, reference)
        
        assert result['is_hallucination'] is True
        assert result['similarity'] < 0.75
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detector.py

# Run with verbose output
pytest -v
```

## 📋 Pull Request Process

### Before Submitting

1. **Update Documentation:** Ensure README and docstrings are updated
2. **Add Tests:** Include tests for new features
3. **Run Tests:** All tests must pass
4. **Format Code:** Run Black and isort
5. **Update CHANGELOG:** Add entry describing changes

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new features
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented if unavoidable)

## Related Issues
Closes #123
```

### Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer review required
3. Address all review comments
4. Squash commits before merging

## 🆕 Adding New Features

### Adding a New Detection Method

1. Create new file in `src/detection_methods/`
2. Implement base interface:

```python
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, response: str, reference: str) -> Dict:
        pass
```

3. Add tests in `tests/detection_methods/`
4. Update documentation
5. Add example usage in README

### Adding a New Dataset

1. Place dataset in `data/your_dataset/`
2. Create loader in `src/data_loaders/`
3. Follow standard format:

```json
{
  "id": "unique_id",
  "query": "question or prompt",
  "reference": "ground truth",
  "metadata": {}
}
```

4. Add dataset documentation
5. Include citation information

### Adding a New Metric

1. Add method to `src/metrics.py`
2. Include docstring with formula
3. Add unit tests
4. Update metrics documentation

## 🐛 Bug Reports

### Good Bug Report Includes

- **Clear Title:** Descriptive summary
- **Environment:** OS, Python version, package versions
- **Steps to Reproduce:** Minimal example code
- **Expected Behavior:** What should happen
- **Actual Behavior:** What actually happens
- **Error Messages:** Full traceback if applicable

### Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
```python
# Minimal code to reproduce
from src.benchmark import HallucinationBenchmark
benchmark = HallucinationBenchmark(...)
# Steps that cause the bug
```

**Expected behavior**
Description of expected outcome

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- Package version: [e.g., 1.0.0]

**Additional context**
Any other relevant information
```

## 💡 Feature Requests

### Good Feature Request Includes

- **Use Case:** Why is this feature needed?
- **Proposed Solution:** How should it work?
- **Alternatives:** Other approaches considered
- **Additional Context:** Examples, mockups, etc.

## 📚 Documentation

### Documentation Structure

- **README.md:** Overview and quick start
- **docs/:** Detailed documentation
- **examples/:** Example scripts and notebooks
- **API documentation:** Generated from docstrings

### Updating Documentation

```bash
# Build documentation locally
cd docs/
make html

# View documentation
open _build/html/index.html
```

## 🏆 Recognition

Contributors are recognized in:
- README.md Contributors section
- CHANGELOG.md for significant contributions
- GitHub Contributors page

## 📞 Getting Help

- **GitHub Issues:** For bugs and feature requests
- **Discussions:** For questions and general discussion
- **Email:** [maintainer email] for private inquiries

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behaviors:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project team. All complaints will be reviewed and investigated promptly and fairly.

---

Thank you for contributing to making LLM evaluation more rigorous and reliable! 🚀