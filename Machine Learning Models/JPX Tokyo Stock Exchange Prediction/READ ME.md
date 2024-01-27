# Stock Price Prediction Analysis

## Overview
This project aims to predict stock prices using various machine learning models, focusing on ensemble methods and gradient boosting frameworks. By leveraging historical stock price data, the models attempt to forecast future prices, providing insights into potential market trends.

## How It Works
The code performs the following steps:

1. **Data Preprocessing**: Reads stock price data, sorts it by date and security code, and filters based on a specific date to ensure consistency in the data used for modeling.

2. **Feature Engineering**: Enhances the dataset with new features derived from existing ones, such as lagged values of stock prices and volume, weighted volume prices, and various statistical measures like the balance of power and standard deviation of prices. It also includes temporal features derived from the date, such as the day of the week and the month.

3. **Model Training and Evaluation**: Trains four different models - LightGBM, RandomForest, XGBoost, and GradientBoosting - on the engineered features to predict stock prices. Each model's performance is evaluated using Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) metrics.

4. **Optimization**: Implements optimization strategies for RandomForest and GradientBoosting models to improve training efficiency without significantly sacrificing model accuracy.

5. **Results Visualization**: Plots the RMSE and MAE of each model for comparison.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- Matplotlib
- Tqdm

## Running the Code
Ensure all dependencies are installed, and the stock price dataset is available in the same directory as the script. Run the script using a Python interpreter. The output will include the RMSE and MAE for each model and visualizations of their performance.
