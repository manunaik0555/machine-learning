# Checkerboard Classification Project

This project performs exploratory data analysis, data profiling, and classification modeling on a tabular checkerboard dataset using Python, Scikit-Learn, and `ydata-profiling`.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Exploratory Data Analysis & Profiling](#exploratory-data-analysis--profiling)
- [Modeling Pipeline](#modeling-pipeline)
- [Evaluation Results](#evaluation-results)

## Project Overview
The goal of this project is to build and evaluate machine learning models to classify data points based on continuous features. A **Random Forest Classifier** is trained and evaluated using stratified k-fold cross-validation and hyperparameter tuning.

## Dataset
- **Source File:** `/content/checkerboard.csv`
- **Features:** 20 numerical columns (`att1` to `att20`).
- **Target Class:** `class` (categorical labels like `red`, encoded to integer format for processing).

## Requirements
To install the necessary packages for generating the profile reports and plotting, run:
```bash
pip install -U "numpy<2.4.0" ydata-profiling scikit-learn matplotlib pandas
```

## Exploratory Data Analysis & Profiling
We generated an interactive HTML profile report to discover patterns, correlations, and distributions using `ydata-profiling` (a reduced dataset of 5 features was profiled due to Colab constraints). Key feature relationships (e.g., `att3` vs `att4` and `att1` vs `att5`) were visualized via scatter plots colored by class.

## Modeling Pipeline
1. **Data Preprocessing:** 
   - Features extracted to matrices ($X$ and $y$).
   - Target labels mapped to integers using `LabelEncoder`.
2. **Classifier:** `RandomForestClassifier`
3. **Cross-Validation:** Stratified 5-Fold Cross-Validation (`StratifiedKFold`).
4. **Hyperparameter Tuning:** Grid Search (`GridSearchCV`) tuning `max_depth` (`[2,3,4]`) and `n_estimators` (`[100, 200, 500]`) on each fold.

## Evaluation Results
- **Metric Used:** F1 Score (optimized for class imbalance/evaluation).
- **Average Cross-Validated F1 Score:** `~0.495`
- **Area Under Precision-Recall Curve (AUPRC):** `~0.499`
- **Visualizations:** A Precision-Recall Curve is generated to evaluate classifier performance across different thresholds.
