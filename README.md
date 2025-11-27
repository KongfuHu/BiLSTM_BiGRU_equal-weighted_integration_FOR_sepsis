# BiLSTM_BiGRU_equal-weighted_integration_FOR_sepsis
Official implementation of the BiLSTM–BiGRU equal-weighted integration model for infection vs sepsis classification based on protein biomarkers.

This repository contains the official implementation of our deep learning model for distinguishing **infection (0)** from **sepsis (1)** based on **141 circulating protein biomarkers**.  
The model integrates **Bidirectional LSTM (BiLSTM)** and **Bidirectional GRU (BiGRU)** using an **equal-weighted probability ensemble**, providing robust performance under severe class imbalance.

---

## 🔍 Overview
- **Task**: Binary classification — Infection vs Sepsis  
- **Input**: 141 protein features (after preprocessing)  
- **Models**:
  - BiLSTM  
  - BiGRU  
  - Equal-weighted ensemble of both  
- **Metrics**: AUC, Accuracy, Confusion Matrix, Classification Report  
- **Data split**: Train / Validation / Test = 64% / 16% / 20%  
- **Loss function**: `BCEWithLogitsLoss + pos_weight` for imbalance handling  

This model provides a simple, interpretable, and robust sequence-based framework for protein-level sepsis classification.

> **Note**: Real patient data is not included due to privacy restrictions.  
> The code expects input in the same structure as `data_model_141.xlsx`.

---

## 🧬 Data Format

Your dataset should be an Excel file with the following structure:

| eid | protein_1 | protein_2 | ... | protein_141 | sepsis_group |
|-----|-----------|-----------|-----|--------------|---------------|
| 1001 | ... | ... | ... | ... | 0/1 |

- `sepsis_group`: 0 = infection, 1 = sepsis  
- `eid`: optional identifier (not used as a feature)  

The model automatically performs **mean imputation** and **standardization**, so no preprocessing is required.

---

## 🔒 Data Availability and Ethics Statement

The real patient-level protein biomarker datasets (PKUTH and UKB cohorts) used in
this study contain protected health information and are governed by institutional
and national ethical regulations. As such, **the original datasets cannot be shared
openly**.

To support reproducibility, we provide **synthetic dummy datasets** that preserve
the structure and variable definitions of the real data but do not contain any
patient information.

Access to the original datasets requires:
1. A formal data-use request,
2. Approval from the corresponding author's institution,
3. Completion of all necessary ethical review procedures.

Researchers who wish to request access may contact the corresponding author.

---

If you need GPU acceleration, install the corresponding CUDA-enabled PyTorch version from:
https://pytorch.org/get-started/locally/

---

## 🚀 Quick Start

```bash
git clone https://github.com/KongfuHu/BiLSTM_BiGRU_equal-weighted_integration_FOR_sepsis.git
cd BiLSTM_BiGRU_equal-weighted_integration_FOR_sepsis
pip install -r requirements.txt
python train.py
