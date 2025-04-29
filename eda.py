import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway
import numpy as np

def perform_eda(df):
    """
    Performs Exploratory Data Analysis on the housing dataset.

    Args:
        df (pd.DataFrame): The input housing dataset.
    """

    # Descriptive Statistics
    print("Descriptive Statistics:")
    print(df.describe())  #

    # Checking for Null Values
    print("\nNull Value Counts:")
    print(df.isnull().sum())  #

    # Histograms and Boxplots for Numerical Features
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        plot_numerical_distribution(df, col)  # Use the helper function

    # Correlation Matrix
    plot_correlation_matrix(df)  # Use the helper function

    # Scatter Plot: Price vs Area
    plot_price_vs_area(df)  # Use the helper function

    # ANOVA for Categorical Features
    perform_anova_categorical(df)  # Use the helper function

    # Price Bin Analysis
    analyze_price_bins(df)  # Use the helper function

    # ANOVA for Numerical Features vs. Price Bin
    perform_anova_price_bins(df)  # Use the helper function


def plot_numerical_distribution(df, col):
    """Helper function to plot histogram and boxplot for a numerical column."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize for better layout
    sns.boxplot(y=df[col], ax=axes[0])
    axes[0].set_title(f'Boxplot of {col}')
    sns.histplot(df[col], kde=True, color='red', bins=30, ax=axes[1])
    axes[1].set_title(f'Histogram of {col}')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """Helper function to plot the correlation matrix."""

    numeric_df = df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

def plot_price_vs_area(df):
    """Helper function to plot Price vs Area scatter plot and with regression line."""

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='area', y='price', data=df, color='green')
    plt.title('Price vs Area')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.show()

    sns.lmplot(x='area', y='price', data=df, aspect=1.5, height=6, line_kws={'color': 'red'})
    plt.title('Price vs Area with Regression Line')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.show()

def perform_anova_categorical(df):
    """Helper function to perform ANOVA tests for categorical features."""

    categorical_features = ['mainroad', 'guestroom', 'basement',
                            'hotwaterheating', 'airconditioning', 'prefarea',
                            'furnishingstatus']  #
    print("\nANOVA Results for Categorical Features:")
    for feature in categorical_features:
        groups = [df[df[feature] == category]['price'] for category in df[feature].unique()]
        stat, p = f_oneway(*groups)
        print(f"ANOVA for {feature}: F-statistic: {stat:.3f}, p-value: {p:.4f}")  #

def analyze_price_bins(df):
    """Helper function to analyze the 'price_bin' feature."""

    print("\nPrice Bin Analysis:")
    print("Value Counts:\n", df['price_bin'].value_counts())  #
    print("\nProportions:\n", df['price_bin'].value_counts(normalize=True))  #

    plt.figure(figsize=(8, 6))
    sns.countplot(x='price_bin', data=df, palette='Set2')  #
    plt.title('Count of price_bin')
    plt.xlabel('price_bin')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(6, 6))
    df['price_bin'].value_counts().plot.pie(autopct='%1.1f%%',
                                            colors=['lightblue', 'lightgreen', 'salmon'])  #
    plt.title('price_bin Distribution')
    plt.ylabel('')
    plt.show()

    numerical_cols = ['area', 'bedrooms', 'bathrooms']  #
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='price_bin', y=col, data=df, palette='Pastel1')  #
        plt.title(f'{col.capitalize()} by price_bin')
        plt.xlabel('price_bin')
        plt.ylabel(col.capitalize())
        plt.show()

    sns.pairplot(df, hue='price_bin', vars=['price', 'area', 'bedrooms',
                                        'bathrooms'], palette='husl')  #
    plt.suptitle('Pairplot by price_bin', y=1.02)
    plt.show()

def perform_anova_price_bins(df):
    """Helper function to perform ANOVA tests for numerical features vs. 'price_bin'."""

    features = ['area', 'bedrooms', 'bathrooms', 'parking']  #
    print("\nANOVA Results for Numerical Features vs. Price Bin:")
    for feature in features:
        groups = [df[df['price_bin'] == status][feature] for status in
                  df['price_bin'].unique()]
        stat, p = f_oneway(*groups)
        print(f"ANOVA for {feature} - F-statistic: {stat:.3f}, p-value: {p:.4f}")  #


if __name__ == '__main__':
    # Create a dummy DataFrame for testing
    import numpy as np
    data = {'price': [100000, 200000, 300000, 400000, 500000],
            'area': [1000, 2000, 3000, 4000, 5000],
            'bedrooms': [1, 2, 3, 4, 5],
            'bathrooms': [1, 1, 2, 2, 3],
            'stories': [1, 2, 2, 3, 4],
            'mainroad': ['yes', 'no', 'yes', 'no', 'yes'],
            'guestroom': ['no', 'yes', 'no', 'no', 'yes'],
            'basement': ['no', 'no', 'yes', 'yes', 'no'],
            'hotwaterheating': ['no', 'no', 'no', 'yes', 'yes'],
            'airconditioning': ['no', 'no', 'yes', 'yes', 'yes'],
            'parking': [0, 1, 2, 2, 3],
            'prefarea': ['yes', 'no', 'yes', 'no', 'no'],
            'furnishingstatus': ['unfurnished', 'furnished', 'semi-furnished', 'furnished', 'unfurnished'],
            'price_bin': ['Low', 'Low', 'Medium', 'Medium', 'High']}
    df = pd.DataFrame(data)

    perform_eda(df)
    print("\nEDA completed!")