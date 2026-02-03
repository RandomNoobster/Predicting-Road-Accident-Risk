import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_frequencies(df):
    cols = df.columns
    n_rows = (len(cols) + 2) // 3  # Rounds up to fit all columns
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, col in enumerate(cols):
        plt.subplot(n_rows, 3, i + 1)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], kde=False)
        else:
            sns.countplot(data=df, x=col)
            
        plt.title(col)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()