import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def print_evaluate(true, predicted):
    """
    Prints regression evaluation metrics.

    Args:
        true:  True target values.
        predicted: Predicted target values.
    """
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    mape = metrics.mean_absolute_percentage_error(true, predicted)
    r2_score = metrics.r2_score(true, predicted)

    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('MAPE:', mape)
    print('R^2:', r2_score)

def evaluate(true, predicted):
    """
    Calculates and returns regression evaluation metrics.

    Args:
        true:  True target values.
        predicted: Predicted target values.

    Returns:
        tuple: (MAE, MSE, RMSE, MAPE, R^2)
    """
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    mape = metrics.mean_absolute_percentage_error(true, predicted)
    r2_score = metrics.r2_score(true, predicted)
    return mae, mse, rmse, mape, r2_score

def train_and_evaluate_regression_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates several regression models.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target.
        y_test: Testing target.
    """

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    test_pred_lin = lin_reg.predict(X_test)
    print('\nLinear Regression:')
    print('Test set evaluation:')
    print_evaluate(y_test, test_pred_lin)

    results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred_lin)]],
                              columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R^2'])

    # Decision Tree Regression
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_train)
    test_pred_dt = dt_reg.predict(X_test)
    print('\nDecision Tree Regression:')
    print('Test set evaluation:')
    print_evaluate(y_test, test_pred_dt)

    results_df_dt = pd.DataFrame(data=[["Decision Tree Regression", *evaluate(y_test, test_pred_dt)]],
                                 columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R^2'])
    results_df = pd.concat([results_df, results_df_dt], ignore_index=True)

    # Random Forest Regression
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)
    test_pred_rf = rf_reg.predict(X_test)
    print('\nRandom Forest Regression:')
    print('Test set evaluation:')
    print_evaluate(y_test, test_pred_rf)

    results_df_rf = pd.DataFrame(data=[["Random Forest Regression", *evaluate(y_test, test_pred_rf)]],
                                 columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R^2'])
    results_df = pd.concat([results_df, results_df_rf], ignore_index=True)

    print("\nModel Comparison:")
    print(results_df)

    # Visualization (example for Random Forest)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test[0:50])), y_test[0:50], "r-", label="Actual values")
    plt.plot(np.arange(len(test_pred_rf[0:50])), test_pred_rf[0:50], "g-", label="Predicted values")
    plt.scatter(np.arange(len(y_test[0:50])), y_test[0:50], label="Actual values")
    plt.scatter(np.arange(len(test_pred_rf[0:50])), test_pred_rf[0:50], label="Predicted values")
    plt.legend(['Actual', 'Predicted'])
    plt.title('Actual vs Predicted of first 50 data points for test data (Random Forest)')
    plt.show()

if __name__ == '__main__':
    # Create dummy data for testing
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [11, 12, 10, 14, 15, 16, 13, 18, 19, 20],  # Changed feature2
        'target': [102, 198, 305, 395, 503, 597, 701, 799, 904, 1001]  # Changed target
    }
    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_and_evaluate_regression_models(X_train, X_test, y_train, y_test)