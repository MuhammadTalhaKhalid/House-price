House Price Classification & Regression Analysis
This project focuses on analyzing, visualizing, and modeling housing price data. The dataset is sourced from Kaggle and is used to perform both classification (categorizing prices as Low, Medium, or High) and regression (predicting actual price values) using machine learning techniques.

📁 Dataset
Source: Kaggle - Housing Prices Dataset

File Used: Housing.csv

Target Variables:

price (for regression)

price_bin (for classification - Low, Medium, High)

📌 Key Features of the Project
📊 Data Exploration and Visualization
Summary statistics and data inspection

Distribution plots, boxplots, heatmaps, pairplots

Correlation matrix

Categorical analysis using ANOVA

🛠️ Feature Engineering
Created price_bin based on quantiles

Added total_rooms feature (sum of bedrooms, bathrooms, stories)

Label encoding of categorical features

🧹 Data Preprocessing
Missing value analysis

Standardization using StandardScaler

Applied SMOTE to handle class imbalance in classification

🤖 Models Used
For Classification (Predicting Price Category):
Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

Support Vector Machine (SVM)

For Regression (Predicting Actual Price):
Linear Regression

Decision Tree Regressor

Random Forest Regressor



📈 Evaluation Metrics
Classification:
Accuracy

Precision, Recall, F1-Score

Confusion Matrix

ROC-AUC Score

Regression:
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

