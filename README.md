
# Multi-Label Cancer Biomarker Analysis and Prediction Models

This repository contains Python code and resources for analyzing cancer biomarkers and predicting outcomes using multi-label classification models, survival analysis, and mutational signature generation. These tools aim to explore the relationships between various biomarkers and their predictive power across different cancer types.

## Table of Contents

1. Overview
2. Features
3. Repository Structure
4. Requirements
5. Usage
6. Results
7. License

## Overview

This project uses machine learning and statistical analysis techniques to study cancer biomarkers and generate insights into their roles in different cancers. Key components include multi-label classification, survival analysis, and mutational signature profiling.

## Features

- **Multi-label classification models:** Train deep learning models for identifying cancer biomarkers.
- **Visualization:** Generate and compare biomarker prediction accuracies using box plots.
- **Survival Analysis:** Perform mixture model analysis for survival prediction.
- **Mutational Signature Analysis:** Generate and evaluate mutational signatures using SigProfilerExtractor.

## Repository Structure

```
├── mlc_model.py
│   - Multi-label classification model for cancer biomarker prediction.
│
├── validation.py
│   - Validation scripts for analyzing prediction accuracy of biomarker groups.
│
├── survival_code.py
│   - Code for survival analysis using Gaussian Mixture Models.
│
├── Mutational_signature_generation.py
│   - Code for generating mutational signatures using SigProfilerExtractor.
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

2. **Validation**  
   Use `validation.py` to generate validation plots for prediction accuracies.
   ```bash
   python validation.py
   ```

3. **Survival Analysis**  
   Run `survival_code.py` for Gaussian Mixture Model-based survival analysis.
   ```bash
   python survival_code.py
   ```

4. **Mutational Signature Generation**  
   Use `Mutational_signature_generation.py` to generate mutational signatures.
   ```bash
   python Mutational_signature_generation.py
   ```

## Results

- Multi-label classification results include classification reports and accuracy metrics for cancer biomarkers.
- Visualization includes box plots comparing prediction accuracies across biomarker groups.
- Survival analysis outputs Gaussian Mixture Models for survival prediction.
- Mutational signature profiling provides extracted signatures and their interpretations.


