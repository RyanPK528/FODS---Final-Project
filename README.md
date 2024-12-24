# FODS---Final-Project

BINUS International - Fundamental of Data Science

Members:
1. Matthew Staniswinata
2. Steven Gerald Marlent
3. Ryan Patrick Komala

## Project Overview

This project is part of the Fundamental of Data Science course at BINUS International. The goal of this project is to analyze job market data and build various machine learning models to predict salaries based on job descriptions and other features.

## Project Structure

```
FODS---Final-Project
├─ Cleaned_data
│  └─ salary_data_processed.csv
├─ DecisionTree.py
├─ GradeBoosting.py
├─ Knn_Algo.py
├─ LinearRegression.py
├─ README.md
├─ RandomForest.py
└─ Raw data
   ├─ Indeed.csv
   ├─ cleaned_kalibrr_data.csv
   ├─ jobstreet_data.csv
   ├─ loker_cleaned.csv
   └─ merged_jobs.csv
```

- **Cleaned_data**: Contains processed data files used for training and testing the models.
  - `salary_data_processed.csv`: Processed salary data.

- **Raw data**: Contains raw data files collected from various job portals.
  - `cleaned_kalibrr_data.csv`
  - `Indeed.csv`
  - `jobstreet_data.csv`
  - `loker_cleaned.csv`
  - `merged_jobs.csv`

- `DecisionTree.py`: The decision tree model for this final project
- `GradeBoosting.py`: The gradient descent learning algorithm for this final project
- `Knn_Algo.py`: Main algorithm for this final project
- `LinearRegression.py`: Base algorithm for this final project
- `RandomForest.py`: Second best-fit algorithm for this final project

## How to Run

1. Ensure you have Python installed on your system.
2. Install the required libraries using the following command:
   ```sh
   pip install -r requirements.txt