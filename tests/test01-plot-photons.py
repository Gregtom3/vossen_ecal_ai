import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_features(df, dataset_name, output_dir):
    """
    For each numeric column in the DataFrame, plot a histogram and save it.
    """
    # Create directory for dataset plots if it doesn't exist
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    for col in df.columns:
        # Check if column is numeric; if not, skip it.
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure()
            df[col].plot(kind='hist', bins=30, edgecolor='black', title=f'{dataset_name} - {col}')
            plt.xlabel(col)
            plt.ylabel('Counts')
            # Save plot as PNG file
            filename = os.path.join(dataset_dir, f'{col}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot for {dataset_name} feature '{col}' to {filename}")
        else:
            print(f"Skipping non-numeric feature '{col}' in {dataset_name}.")

def main():
    train_csv = "../data/photon_data_train.csv"
    test_csv  = "../data/photon_data_test.csv"
    val_csv   = "../data/photon_data_val.csv"

    # Read CSV files into DataFrames
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    val_df   = pd.read_csv(val_csv)
    
    # Define output directory for plots
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot features for each CSV
    plot_features(train_df, "train", output_dir)
    plot_features(test_df, "test", output_dir)
    plot_features(val_df, "val", output_dir)

if __name__ == "__main__":
    main()
