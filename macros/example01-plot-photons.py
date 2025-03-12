import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math

def plot_features(df, dataset_name, output_dir):
    """
    Make an N x 5 grid of plots for the numeric features in the DataFrame.
    Each subplot shows a histogram of one feature.
    """
    # Create directory for dataset plots if it doesn't exist
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Get a list of numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Determine grid dimensions (5 columns)
    n_cols = 5
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    
    # Create a figure and axes for the grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    
    # If there is only one row and one column, make sure axs is iterable
    if n_rows * n_cols == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # Plot each numeric feature in its corresponding subplot
    for ax, col in zip(axs, numeric_cols):
        df[col].plot(kind='hist', bins=30, edgecolor='black', ax=ax, title=col)
        ax.set_xlabel(col)
        ax.set_ylabel('Counts')
    
    # Hide any unused subplots
    for ax in axs[len(numeric_cols):]:
        ax.set_visible(False)
    
    # Set a super title for the grid
    fig.suptitle(f'{dataset_name.capitalize()} Features', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    filename = os.path.join(dataset_dir, 'features.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved {dataset_name} features plot grid to {filename}")


def main():
    train_csv = "data/false_photons/photon_data_train.csv"
    test_csv  = "data/false_photons/photon_data_test.csv"
    val_csv   = "data/false_photons/photon_data_validation.csv"

    # Read CSV files into DataFrames
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    val_df   = pd.read_csv(val_csv)
    
    # Define output directory for plots
    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)

    # Plot features for each CSV
    plot_features(train_df, "train", output_dir)
    plot_features(test_df, "test", output_dir)
    plot_features(val_df, "val", output_dir)

if __name__ == "__main__":
    main()
