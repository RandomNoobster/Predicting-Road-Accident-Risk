import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def replace_categorical(df):
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
    
def normalize(train_df, val_df, test_df, columns, scaler):    
    train_df[columns] = scaler.fit_transform(train_df[columns])
    val_df[columns] = scaler.transform(val_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])
    
    return train_df, val_df, test_df


def mess_up_selected(df, exclude_cols, pct=0.05):
    target_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Apply noise
    for col in target_cols:
        mask = np.random.rand(len(df)) < pct
        df.loc[mask, col] = np.nan
        
    return df

def impute_data(X_train, X_val, X_test, continuous_cols, cont_strategy="mean", cat_strategy="mode"):
    xt_c, xv_c, xe_c = X_train.copy(), X_val.copy(), X_test.copy()
    
    # Categorical columns are those not defined as continuous
    categorical_cols = [c for c in xt_c.columns if c not in continuous_cols]

    def apply_logic(df_train, df_val, df_test, cols, strategy):
        for col in cols:
            if strategy == "mean":
                fill = df_train[col].mean()
            elif strategy == "median":
                fill = df_train[col].median()
            elif strategy == "mode":
                # mode() returns a Series, so we take the first value
                modes = df_train[col].mode()
                fill = modes.iloc[0] if not modes.empty else np.nan
            elif strategy == "random":
                train_values = df_train[col].dropna().values
                if len(train_values) > 0:
                    for df in [df_train, df_val, df_test]:
                        mask = df[col].isna()
                        if mask.any():
                            df.loc[mask, col] = np.random.choice(train_values, size=mask.sum())
                continue # Random is handled row-by-row, so skip the fillna block below
            else:
                raise ValueError(f"Not a valid strategy: {strategy}")
            
            df_train[col] = df_train[col].fillna(fill)
            df_val[col] = df_val[col].fillna(fill)
            df_test[col] = df_test[col].fillna(fill)
            
        return df_train, df_val, df_test

    # Process Continuous and Categorical
    xt_c, xv_c, xe_c = apply_logic(xt_c, xv_c, xe_c, continuous_cols, cont_strategy)
    xt_c, xv_c, xe_c = apply_logic(xt_c, xv_c, xe_c, categorical_cols, cat_strategy)

    return xt_c, xv_c, xe_c