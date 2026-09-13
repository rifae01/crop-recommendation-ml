# Crop Recommendation System Using Machine Learning

A Machine Learning based Crop Recommendation System developed using Python. The system analyzes soil and environmental parameters and recommends a suitable crop based on the trained Machine Learning model.

## Project Overview

The Crop Recommendation System uses Machine Learning classification algorithms to predict a suitable crop based on the following parameters:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The project demonstrates the complete Machine Learning workflow, including data loading, preprocessing, data analysis, visualization, model training, model evaluation, model comparison, and crop prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code

## Machine Learning Algorithms

The following classification algorithms are implemented and compared:

1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)

The model with the best accuracy is selected for the final crop prediction.

## Dataset

The dataset contains soil and environmental parameters along with the corresponding crop label.

### Features

| Feature | Description |
|---|---|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| Temperature | Temperature of the environment |
| Humidity | Relative humidity |
| pH | Soil pH value |
| Rainfall | Rainfall measurement |
| Label | Recommended crop |

## Machine Learning Workflow

Dataset
   ↓
Data Understanding
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature and Target Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Crop Prediction
