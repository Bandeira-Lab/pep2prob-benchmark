# Pep2Prob-benchmark

This repo reproduces experiments from  
**"Pep2Prob Benchmark: Predicting Fragment Ion Probability for MS<sup>2</sup>-based Proteomics.”**

the training script is `run.py`, located in the root directory. It supports training and evaluating various models on the Pep2Prob benchmark dataset. It can be run with models: ..., on split index from 1 to 5, with different training parameters.

## Installation
To run the training script, you need to set up your environment with the required dependencies. You can use the provided `environment_gpu.yml` file to create a conda environment. 
If you are using conda, you can create the environment with the following command:

```bash
conda env create -f environment_gpu.yml
```

If you are using pip, you can install the required packages with:
```bash
pip install -r requirement_gpu.txt
```

We also provide a `requirement_cpu.txt` file for CPU-only environments. You can install the required packages with:
```bash
pip install -r requirement_cpu.txt
```

Here we list the main dependencies for the environment we use for running experiments reported in the paper as a reference.

```yaml
python == 3.11
numpy == 1.25.2
pandas == 2.0.3
pytorch == 2.4.0
pytorch-cuda == 12.4
pyarrow == 19.0.0      # for reading parquet files
```
