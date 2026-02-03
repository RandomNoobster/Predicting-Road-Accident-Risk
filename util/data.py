import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    cleaned = df.drop(columns=['id'])
    return cleaned

def split_data(df, target_col):
    X = df.drop(columns=[target_col])
    Y = df[target_col]
    return X, Y

def train_val_test_split(X, Y, test_size=0.1, val_size=0.1, random_state=42):
    relative_val_size = val_size / (1 - test_size)

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=relative_val_size, random_state=random_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test