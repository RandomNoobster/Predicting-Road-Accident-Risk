import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    cleaned = df.drop(columns=['id'])

    # One hot encode categorical features
    categorical_cols = cleaned.select_dtypes(include=['object', 'category']).columns
    cleaned = pd.get_dummies(cleaned, columns=categorical_cols, drop_first=True)
    
    # Turn booleans into numbers
    bool_cols = cleaned.select_dtypes(include=['bool']).columns
    cleaned[bool_cols] = cleaned[bool_cols].astype(int)
    
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

def summarize_columns(df):
    for col in df.columns:        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            print(f"{col}: Range: {col_min} to {col_max}")
        else:
            unique_values = df[col].unique()
            print(f"{col}: Categories: {list(unique_values)}")
    
def normalize(train_df, val_df, test_df, columns):
    scaler = MinMaxScaler()
    
    train_df[columns] = scaler.fit_transform(train_df[columns])
    val_df[columns] = scaler.transform(val_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])
    
    return train_df, val_df, test_df