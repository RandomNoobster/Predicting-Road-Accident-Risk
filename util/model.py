from sklearn.linear_model import Lasso
import joblib

# DUPLICATE from main notebook
from util.data import load_data, clean_data, split_data, train_val_test_split, summarize_columns, normalize
from util.visualizations import plot_frequencies

# DUPLICATE from main notebook
def prepare_data():
    TRAIN_DATA = "data/train.csv"
    TEST_DATA  = "data/test.csv"
    TARGET_COL = "accident_risk"
    data = load_data(TRAIN_DATA)

    data_cleaned = clean_data(data)

    X, Y = split_data(data_cleaned, TARGET_COL)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = train_val_test_split(X, Y)

    X_train, X_val, X_test = normalize(X_train, X_val, X_test, ["speed_limit"])
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def train_baseline_model(X_train, y_train):
    """
    Train a baseline regression model using Lasso regression. Save the trained model.
    """
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, "models/baseline_model.joblib")
    
    return model

def get_baseline_model():
    """
    Regression
    """
    # Load the model if it exists
    try:
        model = joblib.load("models/baseline_model.joblib")
        return model
    except FileNotFoundError:
        pass
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    model = train_baseline_model(X_train, y_train)
    return model

def tree_model():
    """
    Tree
    """

    pass

def final_model():
    """
    Advanced ???
    """


    pass