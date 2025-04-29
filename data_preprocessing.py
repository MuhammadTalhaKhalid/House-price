import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

def preprocess_data(df):
    """
    Performs feature engineering and preprocessing on the housing dataset.

    Args:
        df (pd.DataFrame): The input housing dataset.

    Returns:
        tuple: A tuple containing the processed data: 
               (X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y2_train, y2_test).
    """

    # Feature Engineering
    df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['stories']

    # Label Encoding for object type columns
    obj = df.select_dtypes(include='object')
    encoder = LabelEncoder()
    for feature in obj:
        df[feature] = encoder.fit_transform(df[feature])

    # Binning Price into categories
    low = df['price'].quantile(0.25)
    high = df['price'].quantile(0.75)
    bins = [0, low, high, df['price'].max()]
    labels = ['Low', 'Medium', 'High']
    df['price_bin'] = pd.cut(df['price'], bins=bins, labels=labels, include_lowest=True)

    # Separating features and target variables
    X = df.drop(['price_bin', 'price'], axis=1)
    y = df['price_bin']  # for classification
    y2 = df['price']     # for regression

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_reg, X_test_reg, y2_train, y2_test = train_test_split(X, y2, test_size=0.3, random_state=42)

    # Scaling numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    # Handling class imbalance with SMOTE
    print("Class distribution before SMOTE:", Counter(y_train))
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:", Counter(y_train))

    return X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y2_train, y2_test

if __name__ == '__main__':
    # Load the data (you might need to adjust the path)
    df = pd.read_csv('Housing.csv')  # Assuming Housing.csv is in the same directory

    # Preprocess the data
    X_train, X_test, y_train, y_test, X_train_reg, X_test_reg, y2_train, y2_test = preprocess_data(df)

    print("\nPreprocessing completed. Here's a glimpse of the processed data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)