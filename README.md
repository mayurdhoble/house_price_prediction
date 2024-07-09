## House Price Prediction

This project predicts house prices based on various independent features like number of rooms, age of house, number of bedrooms, and salary income. It leverages Flask for a user-friendly web interface and explores a range of machine learning models for prediction.

Features:

Multiple Model Options: Explore different models (Linear Regression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor, Random Forest Regressor, SVR, KNN Regressor, LightGBM, XGBoost) and compare their performance.
Flask Front-End: Interact with the model easily through a web interface built with Flask. Input house features and get predicted prices.
Preprocessing Flexibility: (Optional) Implement data cleaning, scaling, and feature engineering techniques to potentially improve model performance (code for these steps might not be included initially).

Models Used:

1) Linear Regression: A classic model that finds a linear relationship between features and price. 

2) Ridge Regression: Similar to Linear Regression, but penalizes large coefficients to reduce overfitting. 

3) Lasso Regression: Like Ridge, but uses L1 regularization, potentially setting some coefficients to zero for even greater feature selection. (Can be effective with sparse data)

4) ElasticNet: Combines Ridge and Lasso regularizations, offering flexibility in feature selection.
 
5) SGDRegressor: Uses Stochastic Gradient Descent for linear regression, suitable for large datasets.

6) HuberRegressor: Similar to least squares regression, but less sensitive to outliers. (Robust to noise)
  
7) Random Forest Regressor: Ensemble method that combines multiple decision trees for improved accuracy and robustness.
 
8) SVR (Support Vector Regression): Uses support vectors to learn a non-linear relationship between features and price.
  
9) KNeighborsRegressor: Predicts price based on the k nearest neighbors in the training data with similar features.
  
10) LightGBM (Light Gradient Boosting Machine): Powerful gradient boosting algorithm known for speed and accuracy.
 
11) XGBoost (eXtreme Gradient Boosting): Another powerful gradient boosting algorithm offering various tuning options.

Requirements:

Python 3.x (https://www.python.org/downloads/)

Flask (https://flask.palletsprojects.com/en/2.2.x/installation/)

scikit-learn (https://scikit-learn.org/stable/)

NumPy (https://numpy.org/)

Pandas (https://pandas.pydata.org/)

Installation:

Usage:

Data Preparation:

You can use provided clean dataset with house features and corresponding prices.

Model Training:

The code likely includes sections for model selection, training, and evaluation.

Follow the instructions within the code to train and test the various models on your data.

Flask Application:

The code might have a separate Flask app for the web interface.

Follow the instructions to run the Flask app and access the prediction interface in your web browser.

Additional Notes:

Experiment with different models and hyperparameter tuning to optimize prediction accuracy.

To potentially improve model performance (may require additional code).

Choose the model that best suits for you specific dataset and requirements in terms of accuracy, interpretability, and training speed.
