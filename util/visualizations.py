import math
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

def plot_all_features(X, y):
    num_features = X.shape[1]
    cols_per_row = 3
    num_rows = math.ceil(num_features / cols_per_row)
    
    # Create the grid
    fig, axes = plt.subplots(
        nrows=num_rows, 
        ncols=cols_per_row, 
        figsize=(18, 5 * num_rows)
    )
    
    # Flatten axes array for easy 1D iteration
    axes_flat = axes.flatten()

    for i, col in enumerate(X.columns):
        hb = axes_flat[i].hexbin(
            X[col], y, 
            gridsize=35, 
            cmap='inferno', 
            mincnt=1
        )
        
        plt.colorbar(hb, ax=axes_flat[i])
        
        axes_flat[i].set_title(f'{col} Density')
        axes_flat[i].set_xlabel(col)
        axes_flat[i].set_ylabel('Target')

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()