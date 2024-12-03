# House Price Prediction Model for Ames, Iowa

This project aims to develop a machine learning model to predict house sale prices in Ames, Iowa. The model is validated using the Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.

## Table of Contents
- [Introduction](#introduction)
- [Data Processing](#data-processing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Prediction](#prediction)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
The goal of this project is to predict house sale prices using various features from the dataset provided by Kaggle. The model is trained and validated using the RMSE metric.

## Data Processing
The data processing steps include:
1. Reading the training and testing datasets.
2. Handling missing values by filling them with the mean for numerical columns and the most frequent value for categorical columns.
3. Encoding categorical features using LabelEncoder.

## Feature Engineering
New features are created to improve the model's performance:
- **TotalSF**: Sum of the first floor, second floor, and basement square footage.
- **Age**: Difference between the year sold and the year built.
- **LogLotArea**: Log transformation of the lot area to reduce skewness.

## Model Training
The model is trained using the XGBoost algorithm. The training data is split into training and validation sets to prevent overfitting. The model is trained using the following parameters:
```python
params = {
  'objective': 'reg:squarederror', 
  'eval_metric': 'rmse',          
  'learning_rate': 0.05,           
  'max_depth': 6,                  
  'subsample': 0.8,               
  'colsample_bytree': 0.8,         
  'seed': 42                    
}
```
## Hyperparameter Tuning
GridSearchCV is used to find the best hyperparameters for the model. The parameter grid includes:
```python
param_grid = {
  'max_depth': [3, 5, 7],
  'learning_rate': [0.01, 0.05, 0.1],
  'subsample': [0.6, 0.8, 1.0],
  'colsample_bytree': [0.6, 0.8, 1.0],
  'n_estimators': [100, 500, 1000]
}
```
## Model Evaluation
The model is evaluated using the RMSE metric on the validation set. The RMSE is also converted back to the original scale to understand the approximate error in sale prices.

## Feature Importance
XGBoost provides a way to visualize the importance of features used in the model. The top 20 features are plotted to understand their impact on the predictions.

## Prediction
The model is used to predict the sale prices for the test dataset. The predictions are converted back to the original scale and saved to a CSV file for submission.

## Conclusion
This project demonstrates the process of developing a machine learning model to predict house sale prices. The model is trained and validated using various features and hyperparameters to achieve the best performance.

## References
- [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

Feel free to explore the code and modify it to improve the model's performance. Happy coding!