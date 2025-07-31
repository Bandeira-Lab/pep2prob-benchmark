# **Pep2Prob Benchmark**

> **Pep2Prob Benchmark** provides code and data to reproduce the experiments from  
> *“Pep2Prob Benchmark: Predicting Fragment Ion Probability for MS²-based Proteomics”*.

---

## 📖 Overview

Tandem mass spectrometry (MS²) identifies peptides by fragmenting precursor ions and measuring resulting spectra.  
Accurate **fragment ion probability** models are crucial for downstream tasks, including database search, spectral library matching, de novo sequencing and other tools for peptide identification and quantification from MS² data.  

Our **Pep2Prob Benchmark** provides:

- The first curated dataset **Pep2Prob** contains peptide-specific fragment probabilities, where each precursor (peptide sequence, charge state) has a vector showing the probabilities of a list of appearing fragment ions for such precursor.  
- A **train-test split** method to prevent data leakage.
- A **standardized benchmark** with five baseline methods of increasing capacity: Global model, Bag of Fragment ion model, Linear regression model, Resnet, and a transformer-type model. We train these models in the Pep2Prob dataset to predict and evaluate the probability statistics of given precursors.
 

---

## 🗃️ Dataset

- **610,117 unique precursors** (peptide sequence + charge)  
- Constructed from **183 million high-resolution HCD MS² spectra** from 227 mass spectrometry datasets in the MassIVE repository.
- **235 possible fragment ions** per precursor (b-, y-ions with up to 3 charges, as well as a-ions with charge 1 at position 2)  
- **Probability vectors** \($p(f|p)\in[0,1]^{235}$\) estimated by counting the presence of the fragment ions given the precursor across repeated spectra  
- **Train/test split** avoids leakage by grouping similar sequences (identical, shared 6-mer prefix/suffix) into disjoint folds 

---

## ⚙️ Benchmark

We evaluate FIVE methods on Pep2Prob, measuring **$L_1$ loss**, **MSE**, **spectral angle (SA)**, and **fragment-existence metrics**:

| Model         | Capacity            | Test L₁ ↓   | SA ↑    | Existence Accuracy ↑ |
|---------------|---------------------|------------|---------|----------------------|
| **Global**    | global stats only   | 0.244      | 0.558   | 0.699                |
| **BoF**       | + fragment sequence | 0.179      | 0.509   | 0.803                |
| **Linear Reg**| one-hot features    | 0.126      | 0.695   | 0.766                |
| **ResNet**    | 4-layer MLP         | 0.069  | 0.818   | 0.871                |
| **Transformer** | decoder-only       | **0.056**      | **0.845** | **0.953**            |

> Model capacity correlates with better capturing complex sequence-to-fragment relationships.


<!-- | **CNN**       | 4-layer 1d CNN      | 0.072      | 0.808   | 0.870                | -->

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Bandeira-Lab/pep2prob-benchmark.git
   cd pep2prob-benchmark

---
## ⚙️ Usage steps
1. **Set up environment**
   
   You can install the pytorch package with the versions that match your hardware, or use the same environment as mine using the following commands:

  ```shell
  conda create -n pep2prob-env python==3.10
  conda activate pep2prob-env
  pip install -r requirements.txt
  ```

2. **Download dataset & train-test spilt files**
   
   You can use the following commands to download our dataset from Huggingface (https://huggingface.co/datasets/bandeiralab/Pep2Prob). The dataset will be stored in data/pep2prob.

  ```shell
  python data/download_data.py
  ```

3. **Running different baseline models**

You can separately run the following models. The outputs and the final predictions of the models will be saved in the predictions folder.
   *  **_Gloabal model_**
      
      ```shell
      python -u -m models.global.global_model
      ```
   *  **_Bag of Fragment ion model_**
  
      ```shell
      python -u -m models.bag_of_fragment_ion.bof_model
      ```
   *  **_Linear regression model_**
      
      ```shell
      python -u -m models.linear_regression.linear_regression_model
      ```
   *  **_Resnet model_**
      
      ```shell
      python -u -m models.resnet.resnet_model
      ```
   *  **_Transformer model_**
      
      ```shell
      python -u -m models.transformer.transformer_model \
        --precursor_info_path data/pep2prob/pep2prob_dataset.csv \
        --split_path          data/pep2prob/train_test_split_set_1.npy \
        --epochs              2 \
        --batch_size          1024 \
        --lr                  0.001 \
        --weight_decay        0.001 \
        --save_prefix         predictions/transformer_model_run0 \
        --max_length_input    40
       ```
---
