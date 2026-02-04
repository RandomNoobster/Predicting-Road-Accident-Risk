from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import joblib

# DUPLICATE from main notebook
from util.data import load_data, clean_data, split_data, train_val_test_split, summarize_columns, normalize
from util.visualizations import plot_frequencies
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# DUPLICATE from main notebook
def prepare_data():
    TRAIN_DATA = "data/train.csv"
    TEST_DATA  = "data/test.csv"
    TARGET_COL = "accident_risk"
    data = load_data(TRAIN_DATA)

    data_cleaned = clean_data(data)

    X, Y = split_data(data_cleaned, TARGET_COL)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = train_val_test_split(X, Y)

    features_to_normalize = ["speed_limit", "num_lanes"]
    features_to_standardize = ["num_reported_accidents"]

    X_train, X_val, X_test = normalize(X_train, X_val, X_test, features_to_normalize, MinMaxScaler())
    X_train, X_val, X_test = normalize(X_train, X_val, X_test, features_to_standardize, StandardScaler())
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

class BaselineModel:
    """
    Baseline regression model using Random Forest.
    """

    def __init__(self, n_estimators=10, random_state=42):
        # Random Forest is robust and rarely results in "zero importance" for everything
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X) 