# Results Discussion

## Model Performance Overview
The analysis involved four different machine learning models to predict stock prices, and their performance was evaluated based on RMSE and MAE metrics. Below is a summary of each model's performance:

- **LightGBM**
  - RMSE: 0.02580
  - MAE: 0.01841

- **RandomForest**
  - RMSE: 0.02438
  - MAE: 0.01691

- **XGBoost**
  - RMSE: 0.02705
  - MAE: 0.01982

- **GradientBoosting**
  - RMSE: 0.02468
  - MAE: 0.01732

## Analysis
- The **RandomForest** model outperformed the other models in terms of both RMSE and MAE, indicating its effectiveness in capturing the nuances of the stock price data.
- **LightGBM** and **GradientBoosting** showed competitive performance, with GradientBoosting slightly edging out in terms of RMSE.
- **XGBoost** had the highest RMSE and MAE, suggesting it might not have captured the patterns in the data as effectively as the other models in this particular scenario.

## Considerations
- The RandomForest and GradientBoosting models were optimized for training efficiency, which might have influenced their performance metrics.
- The models were trained on a dataset filtered from a specific start date to ensure data consistency, which could impact the generalizability of the models to different time periods or stock conditions.
- The feature engineering process played a crucial role in model performance, highlighting the importance of domain knowledge and creative feature creation in stock price prediction tasks.

## Future Work
- Further hyperparameter tuning could enhance model performance, especially for the XGBoost model, which lagged slightly behind the others.
