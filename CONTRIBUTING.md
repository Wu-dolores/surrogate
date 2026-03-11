# Contributing to Atmospheric Radiation Surrogate Model

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/surrogate.git
   cd surrogate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests** to ensure everything works:
   ```bash
   pytest test_utils.py -v
   ```

## Development Workflow

### 1. Create a Branch

Create a feature branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-attention-mechanism`
- `fix/data-loader-bug`
- `docs/improve-readme`

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add type annotations for new functions
- Include docstrings for all public functions

### 3. Add Tests

Add tests for new functionality in `test_utils.py` or create new test files:

```python
def test_your_new_feature():
    """Test description."""
    # Your test code
    assert result == expected
```

### 4. Run Tests

Before committing, ensure all tests pass:

```bash
# Run all tests
pytest -v

# Run specific test file
pytest test_utils.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add attention mechanism to LocalGNO block

- Implement multi-head attention
- Add configuration options
- Update tests and documentation"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Reference to any related issues
- Screenshots/plots if applicable

## Code Style

### Python Style

Follow PEP 8 guidelines:

```python
# Good
def compute_heating_rate(
    flux: np.ndarray,
    pressure: np.ndarray
) -> np.ndarray:
    """
    Compute heating rate from flux profile.

    Args:
        flux: Net radiative flux (W/m²)
        pressure: Pressure levels (Pa)

    Returns:
        Heating rate (K/day)
    """
    # Implementation
    pass

# Bad
def compute_hr(f,p):
    # no docstring, no types
    pass
```

### Type Annotations

Use type hints for all function signatures:

```python
from typing import Tuple, Optional
import numpy.typing as npt

def process_data(
    x: npt.NDArray[np.float32],
    normalize: bool = True
) -> Tuple[npt.NDArray[np.float32], dict]:
    """Process atmospheric data."""
    pass
```

### Documentation

All public functions should have docstrings:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    Brief description of what the function does.

    Longer description if needed, explaining the algorithm,
    assumptions, or important details.

    Args:
        arg1: Description of first argument
        arg2: Description of second argument

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid

    Example:
        >>> result = function_name(x, y)
        >>> print(result)
    """
    pass
```

## Testing Guidelines

### Unit Tests

Test individual functions in isolation:

```python
def test_normalization():
    """Test z-score normalization."""
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mu, std = zfit(x)
    x_norm = zapply(x, mu, std)

    # Check mean is ~0 and std is ~1
    assert abs(x_norm.mean()) < 1e-5
    assert abs(x_norm.std() - 1.0) < 1e-5
```

### Integration Tests

Test components working together:

```python
def test_data_pipeline():
    """Test complete data loading and preprocessing."""
    loader = AtmosphericDataLoader('test_data.npz')
    raw_data = loader.load_raw_data()
    features = loader.build_features(raw_data)

    assert features.shape[-1] == 7  # Expected feature dimension
```

### Test Data

Use small, synthetic datasets for testing:

```python
@pytest.fixture
def sample_data():
    """Create sample atmospheric data for testing."""
    S, N = 10, 20
    logp = np.linspace(2, 5, N)[None, :].repeat(S, axis=0)
    T = np.random.randn(S, N) * 20 + 250
    # ... create other fields
    return {'logp': logp, 'T': T, ...}
```

## Areas for Contribution

### High Priority

- [ ] Add more unit tests (target: 80% coverage)
- [ ] Implement data augmentation strategies
- [ ] Add TensorBoard logging
- [ ] Performance optimization (vectorization)

### Medium Priority

- [ ] Add attention mechanism option
- [ ] Implement ensemble predictions
- [ ] Create Jupyter notebook examples
- [ ] Add more diagnostic plots

### Documentation

- [ ] Add architecture diagrams
- [ ] Create tutorial notebooks
- [ ] Improve API documentation
- [ ] Add more usage examples

### Nice to Have

- [ ] Web interface for predictions
- [ ] Model compression/quantization
- [ ] Multi-GPU training support
- [ ] Integration with climate model output

## Reporting Issues

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Minimal code to reproduce the bug
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, PyTorch version, OS
6. **Error messages**: Full error traceback if applicable

Example:

```markdown
## Bug: Data loader fails with variable resolution profiles

**Description**: AtmosphericDataLoader crashes when loading profiles with different vertical resolutions.

**Steps to reproduce**:
```python
loader = AtmosphericDataLoader('mixed_resolution.npz')
data = loader.load_raw_data()  # Crashes here
```

**Error**:
```
ValueError: operands could not be broadcast together with shapes (40,) (160,)
```

**Environment**:
- Python 3.9.7
- PyTorch 2.0.1
- macOS 13.0
```

## Questions?

- Open an issue for questions about contributing
- Check existing issues and pull requests first
- Be respectful and constructive in discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
