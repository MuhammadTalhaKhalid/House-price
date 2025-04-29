import pandas as pd
from data_loader import load_housing_data
from data_preprocessing import preprocess_data
from classification_model import train_and_evaluate_classification_models
from regression_model import train_and_evaluate_regression_models


def main():
    # Load the data
    df = load_housing_data()

    # Preprocess the data
    X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y2_train, y2_test = preprocess_data(df)

    # Print class distribution for analysis
    print("Class distribution before preprocessing:")
    print(df['price_bin'].value_counts(dropna=False))

    # Print class distribution after splitting
    print("\nClass distribution in y_train:")
    print(y_train.value_counts())
    print("\nClass distribution in y_test:")
    print(y_test.value_counts())

    # Train and evaluate classification models
    train_and_evaluate_classification_models(X_train, X_test, y_train, y_test)

    # Train and evaluate regression models
    train_and_evaluate_regression_models(X_train_reg, X_test_reg, y2_train, y2_test)


if __name__ == "__main__":
    main()