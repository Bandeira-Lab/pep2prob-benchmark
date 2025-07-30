# **Pep2Prob Benchmark**

> **Pep2Prob Benchmark** provides code and data to reproduce the experiments from  
> *“Pep2Prob Benchmark: Predicting Fragment Ion Probability for MS²-based Proteomics”*.

---

## 📖 Overview

Tandem mass spectrometry (MS²) identifies peptides by fragmenting precursor ions and measuring resulting spectra.  
Accurate **fragment ion probability** models are critical for downstream tasks—database search, spectral library matching, de novo sequencing, PTM localization, and quantification.  

Our **Pep2Prob Benchmark** provides:

- A new curated dataset **Pep2Prob** containing peptide-specific fragment probabilities, where each precusor (peptide sequence+charge state) has a vector showing the probabilities of appearing fragment ions for such precusor.  
- A **train-test split** method.
- A **standardized benchmark** with five baseline methods of increasing capacity: Gloabal model, Bag of Fragment ion model, Linear regression model, Resnet and a transformer-type model. We train these models in Pep2Prob dataset to prediction and evaluate the probability statistics of given precursors.
 

---

## 🗃️ Dataset

- **608 ,780 unique precursors** (peptide sequence + charge)  
- Constructed from **183 ,263 ,674 high-resolution HCD MS² spectra**  
- **235 possible fragment ions** per precursor (a-, b-, y-ions with up to 3 charges)  
- **Probability vectors** \($p(f|p)\in[0,1]^{235}$\) estimated by counting presence of the fragment ions given the precuor across repeated spectra  
- **Train/test split** avoids leakage by grouping similar sequences (identical, shared 6-mer prefix/suffix) into disjoint folds 

---

## ⚙️ Benchmark

We evaluate five methods on Pep2Prob, measuring **$L_1$ loss**, **MSE**, **spectral angle (SA)**, and **fragment-existence metrics**:

| Model         | Capacity            | Test L₁ ↓   | SA ↑    | Existence Accuracy ↑ |
|---------------|---------------------|------------|---------|----------------------|
| **Global**    | global stats only   | 0.244      | 0.558   | 0.699                |
| **BoF**       | + fragment sequence | 0.179      | 0.509   | 0.803                |
| **Linear Reg**| one-hot features    | 0.126      | 0.695   | 0.766                |
| **ResNet**    | 4-layer MLP         | 0.069  | 0.818   | 0.871                |
| **Transformer** | decoder-only       | **0.056**      | **0.845** | **0.953**            |

> Model capacity correlates with better capturing complex sequence-to-fragment relationships. 

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Bandeira-Lab/pep2prob-benchmark.git
   cd pep2prob-benchmark

---
## ⚙️ Usage steps
1. **Set up environment**

  ```shell
  conda create -n pep2prob-env python==3.10
  conda activate pep2prob-env
  pip install -r requirements.txt
  ```

2. **Download dataset & train-test spilt files**

  ```shell
  python data/download_data.py
  ```

3. **Running different baseline models**

You can separately run the following models. The outputs and the final predictions of the models will be saved in the predictions folder.
   1. _Gloabal model_
      ```shell
      python -u -m models.global.global_model
      ```
   2. _Bag of Fragment ion model_
  
      ```shell
      python -u -m models.bag_of_fragment_ion.bof_model
      ```
   3. _Linear regression model_
      ```shell
      python -u -m models.linear_regression.linear_regression_model
      ```
   4. _Resnet_
     ```shell
     python -u -m models.resnet.resnet_model
     ```
   5. _Transformer_
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
