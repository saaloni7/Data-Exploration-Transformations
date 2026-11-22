# Week 4: Data Exploration & Transformations

## 1. Kaggle Dataset Source
- **Dataset:** Medical Cost Personal Datasets
- **Source Link:** https://www.kaggle.com/datasets/mirichoi0218/insurance

## 2. Project Overview
This project performs an exploratory data analysis (EDA) and data transformation study on 1,338 individual healthcare records. Key steps include:
- Cleaning the dataset by removing duplicate records and auditing missing values.
- Applying Tukey's IQR method to detect outliers in numerical attributes (`bmi` and `charges`).
- Engineering features including WHO clinical BMI categories and applying a natural log transformation ($\ln(\text{charges})$) to normalize positive skewness.
- Generating high-resolution visualizations: a distribution histogram (raw vs. log-transformed) and a stratified scatter plot analyzing the interaction between BMI, smoking habits, and medical expenses.

## 3. How to Run
Run the automated cleaning and visualization pipeline:
```bash
python src/data_pipeline.py
```

Launch the interactive Jupyter Notebook:
```bash
jupyter notebook data_exploration_transformations.ipynb
```
