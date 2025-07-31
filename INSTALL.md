# Installation Guide for Pep2Prob

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Options

### Option 1: Install from source (Recommended for development)

```bash
# Clone the repository
git clone https://github.com/bandeiralab/pep2prob-benchmark.git
cd pep2prob-benchmark

# Install in development mode
pip install -e .
```

### Option 2: Install with specific dependencies

```bash
# Install with PyTorch support
pip install -e ".[torch]"

# Install with TensorFlow support  
pip install -e ".[tensorflow]"

# Install with all dependencies
pip install -e ".[all]"

# Install with development tools
pip install -e ".[dev]"
```

### Option 3: Install from requirements.txt

```bash
# Install dependencies first
pip install -r requirements.txt

# Then install the package
pip install -e .
```

## Verify Installation

```bash
# Test the installation
python -c "import pep2prob; print(pep2prob.__version__)"

# Or use the CLI
pep2prob-train --help
```

## Usage Examples

### Using the CLI

```bash
# Train a Transformer model
pep2prob-train --model Transformer --data_split 1 --epochs 100

# Train with custom parameters
pep2prob-train --model ResNN --data_split 2 --epochs 50 --batch_size 512 --learning_rate 0.01
```

### Using as a Python package

```python
from pep2prob import Pep2ProbDataset

# Load dataset
dataset = Pep2ProbDataset(
    data_dir="./data",
    split_idx=1,
    min_length_input=7,
    max_length_input=40,
    min_num_psm_for_statistics=10
)

# Access training and test data
train_data = dataset.train
test_data = dataset.test
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've installed the package with `pip install -e .`

2. **Missing dependencies**: Install missing packages with `pip install -r requirements.txt`

3. **CUDA issues**: If using GPU, ensure PyTorch is installed with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Data download issues**: Ensure you have internet connection and sufficient disk space for the dataset.

### Getting Help

- Check the README.md for more information
- Open an issue on GitHub if you encounter problems
- Ensure all dependencies are properly installed
