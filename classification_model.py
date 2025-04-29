from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_and_evaluate_classification_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates several classification models on the given data.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
    """

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    # Get all unique labels for consistent confusion matrix display
    all_labels = np.unique(np.concatenate([y_train, y_test]))

    for name, model in models.items():
        print(f'Training and evaluating {name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred, zero_division=0))  # Handle zero division

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred, labels=all_labels),  # Use consistent labels
            display_labels=all_labels  # Use consistent labels
        ).plot(cmap=plt.cm.Blues, ax=plt.gca())
        plt.title(f'Confusion Matrix - {name}')
        plt.show()

        # Hyperparameter Tuning (GridSearchCV example for Random Forest)
        if name == 'Random Forest':
            print('Tuning Random Forest...')
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=3,
                scoring='accuracy',
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred_tuned = best_model.predict(X_test)
            accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
            print(f'Tuned Random Forest Accuracy: {accuracy_tuned:.4f}')
            print(classification_report(y_test, y_pred_tuned, zero_division=0))  # Handle zero division

            # Plot Tuned Confusion Matrix
            plt.figure(figsize=(8, 6))
            ConfusionMatrixDisplay(
                confusion_matrix(y_test, y_pred_tuned, labels=all_labels),  # Use consistent labels
                display_labels=all_labels,  # Use consistent labels
            ).plot(cmap=plt.cm.Blues, ax=plt.gca())
            plt.title(f'Tuned Confusion Matrix - Random Forest')
            plt.show()


if __name__ == '__main__':
    # Create dummy data for testing
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'target': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    }
    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2']]
    y = df['target']

    # Adjust test_size if needed based on the number of classes and data size
    if len(y.unique()) > len(y) * 0.2:
        test_size = len(y.unique()) / len(y)
    else:
        test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    train_and_evaluate_classification_models(X_train, X_test, y_train, y_test)