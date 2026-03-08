
# Multi-Label Cancer Biomarker Analysis and Prediction Models

This repository contains Python code and resources for analyzing cancer biomarkers and predicting outcomes using multi-label classification models, survival analysis, mutational signature generation, and baseline model validation. The project focuses on identifying important genomic variants (biomarkers) and evaluating their predictive power across multiple cancer types.
## Overview

This project uses machine learning and statistical analysis techniques to study cancer biomarkers and generate insights into their roles in different cancers. Key components include multi-label classification, survival analysis, baseline model validation and mutational signature profiling.

## Features

- **Multi-label classification models:** Train deep learning models for identifying cancer biomarkers.
- **Visualization:** Generate and compare biomarker prediction accuracies using box plots.
- **Survival Analysis:** Perform mixture model analysis for survival prediction.
- **Mutational Signature Analysis:** Generate and evaluate mutational signatures using SigProfilerExtractor.
- **Baseline model validation:** Compares CLOVER selected varaints against state-of-the-art feature selection approaches. 

## Repository Structure

```
├── mlc_model.py
│   - Multi-label classification model for cancer biomarker prediction.
│
├── survival_code.py
│   - Code for survival analysis using Gaussian Mixture Models.
│
├── Mutational_signature_generation.py
│   - Code for generating mutational signatures using SigProfilerExtractor.
├── DeepFS.py
│   Deep feature selection using supervised autoencoder
│
├── DFS.py
│   Deep Feature Selection implementation using L1 regularization
│
├── HSIC-Lasso.py
│   HSIC-Lasso based feature selection
│
├── LassoNet.py
│   LassoNet implementation for sparse feature selection
│
├── Random_forest.py
│   Feature selection using Random Forest importance
│
└── README.md
    - Project documentation.
```

## Requirements

To set up the environment, ensure you have the following:

- Python 3.8 or later
- Required Python libraries:
  - TensorFlow==2.11.0
  - pandas==1.3.3
  - numpy==1.21.2
  - matplotlib==3.4.3
  - scipy==1.7.1
  - SigProfilerExtractor==1.1.3

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Multi-label Classification**  
   Run `mlc_model.py` to train and evaluate the multi-label classification model.
   ```bash
   python mlc_model.py
   ```

2. **Survival Analysis**  
   Run `survival_code.py` for Gaussian Mixture Model-based survival analysis.
   ```bash
   python survival_code.py
   ```

3. **Mutational Signature Generation**  
   Use `Mutational_signature_generation.py` to generate mutational signatures.
   ```bash
   python Mutational_signature_generation.py
   ```

## Results

- Multi-label classification results include classification reports and accuracy metrics for cancer biomarkers.
- Survival analysis outputs Gaussian Mixture Models for survival prediction.
- Mutational signature profiling provides extracted signatures and their interpretations.


