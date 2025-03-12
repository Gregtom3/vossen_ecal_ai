import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from catboost import CatBoostClassifier
import os

def parse_data(df):
    columns = list(df.columns)
    input_columns = [col for col in columns if col not in ['event', 'y', 'truepid', 'trueparentpid', 'trueparentid']]

    X = df[input_columns]
    y = df['y']
    event = df['event']

    return X, y, event

def train(X_train, X_test, y_train, y_test):
    # Set parameters for the CatBoost model
    params = {
        "iterations": 500,         # Number of iterations (epochs)
        "depth": 4,                # Tree depth
        "learning_rate": 0.1,
        "loss_function": "Logloss",  # For binary classification
        "random_state": 42,
        "verbose": 1             # Controls logging frequency
    }

    # Initialize and train the CatBoost model with early stopping
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test),
            early_stopping_rounds=50)
    
    return model

def loss_plot(model, output_dir):
    evals_result = model.get_evals_result()
    train_loss = evals_result['learn']['Logloss']
    test_loss = evals_result['validation']['Logloss']

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Logloss')
    plt.title('Train vs Test Loss per Iteration')
    plt.legend()
    filename = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved loss plot to {filename}")


def main():
    # Define output directory for plots
    output_dir = "artifacts/example02"
    os.makedirs(output_dir, exist_ok=True)

    # Define csv's with local path
    train_csv = "data/false_photons/photon_data_train.csv"
    test_csv  = "data/false_photons/photon_data_test.csv"
    val_csv   = "data/false_photons/photon_data_validation.csv"


    # Read CSV files into DataFrames
    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)
    df_validation   = pd.read_csv(val_csv)
    
    # Get train, test, validation data
    X_train, y_train, event_train = parse_data(df_train)
    X_test, y_test, event_test = parse_data(df_test)
    X_val, y_val, event_val = parse_data(df_validation)

    # Train and return model
    model = train(X_train, X_test, y_train, y_test)

    # Create loss plot
    loss_plot(model, output_dir)

if __name__ == "__main__":
    main()