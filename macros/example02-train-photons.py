import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from catboost import CatBoostClassifier
import os

def get_mass_vectors(df):

  # =============================================================================
  # Step 1: Define a function to compute the 4-vector for a photon
  # =============================================================================
  def get_4vector(gE, gTheta, gPhi):
      """
      Given a photon's energy (gE), polar angle (gTheta), and azimuthal angle (gPhi),
      calculate its momentum components (px, py, pz) and energy (E).
      """
      px = gE * np.sin(gTheta) * np.cos(gPhi)
      py = gE * np.sin(gTheta) * np.sin(gPhi)
      pz = gE * np.cos(gTheta)
      E  = gE  # For a photon, E is equal to gE (ignoring mass)
      return px, py, pz, E

  # =============================================================================
  # Step 2: Loop over all events to compute invariant masses for different categories
  # =============================================================================
  # Lists to store computed invariant masses:
  all_pi0_masses   = []  # All photon pairs (total)
  subset2_masses   = []  # Both photons: trueparentpid==111 and same trueparentid
  subset3_masses   = []  # Both photons: trueparentpid==111 but different trueparentid
  subset4_masses   = []  # One photon: trueparentpid==111 and one: trueparentpid==-999
  subset5_masses   = []  # Both photons: trueparentpid==-999

  # Get unique event numbers from the DataFrame
  events = df['event'].unique()

  # Loop over each event
  for event_num in events:
      # Filter rows corresponding to the current event
      event_photons = df[df['event'] == event_num]

      # Only consider events with at least two photons
      if len(event_photons) >= 2:
          # Iterate over all unique combinations of two photons in the event
          for photon1, photon2 in itertools.combinations(event_photons.index, 2):
              # Retrieve the data rows for each photon
              row1 = event_photons.loc[photon1]
              row2 = event_photons.loc[photon2]

              # Extract kinematic variables: energy and angles
              gE1, gTheta1, gPhi1 = row1[['gE', 'gTheta', 'gPhi']]
              gE2, gTheta2, gPhi2 = row2[['gE', 'gTheta', 'gPhi']]

              # Compute the 4-vectors for both photons
              px1, py1, pz1, E1 = get_4vector(gE1, gTheta1, gPhi1)
              px2, py2, pz2, E2 = get_4vector(gE2, gTheta2, gPhi2)

              # Sum the momentum and energy to get the total 4-momentum of the pair
              total_px = px1 + px2
              total_py = py1 + py2
              total_pz = pz1 + pz2
              total_E  = E1 + E2

              # Calculate the invariant mass (pi0 candidate mass)
              pi0_mass = np.sqrt(total_E**2 - (total_px**2 + total_py**2 + total_pz**2))
              all_pi0_masses.append(pi0_mass)

              # Extract true parent identifiers for both photons
              tp1 = row1['trueparentpid']
              tp2 = row2['trueparentpid']
              id1 = row1['trueparentid']
              id2 = row2['trueparentid']

              # Categorize the photon pair based on the true parent properties
              # Subset 2: Both photons have trueparentpid==111 and the same trueparentid
              if tp1 == 111 and tp2 == 111 and id1 == id2:
                  subset2_masses.append(pi0_mass)
              # Subset 3: Both photons have trueparentpid==111 but different trueparentid
              elif tp1 == 111 and tp2 == 111 and id1 != id2:
                  subset3_masses.append(pi0_mass)
              # Subset 4: One photon has trueparentpid==111 and the other has trueparentpid==-999
              elif (tp1 == 111 and tp2 == -999) or (tp1 == -999 and tp2 == 111):
                  subset4_masses.append(pi0_mass)
              # Subset 5: Both photons have trueparentpid==-999
              elif tp1 == -999 and tp2 == -999:
                  subset5_masses.append(pi0_mass)
  return all_pi0_masses, subset2_masses, subset3_masses, subset4_masses, subset5_masses

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

def invariant_mass_plot(df_validation, output_dir):

    all_pi0_masses, subset2_masses, subset3_masses, subset4_masses, subset5_masses = get_mass_vectors(df_validation)

    p_threshold = 0.5
    df_validation_copy = df_validation.copy()
    df_validation_copy['p'] = model.predict_proba(X_validation)[:, 1]
    df_validation_copy = df_validation_copy[df_validation_copy['p'] > p_threshold]

    all_pi0_masses_trained, subset2_masses_trained, subset3_masses_trained, subset4_masses_trained, subset5_masses_trained = get_mass_vectors(df_validation_copy)
    # Define xlimits and bins
    xlimits = (0.06, 0.55)
    bins = 100

    # Define mass window limits for counting events
    mass_min, mass_max = 0.106, 0.166

    # Compute counts for the Validation Set
    total_count = np.sum((np.array(all_pi0_masses) > mass_min) & (np.array(all_pi0_masses) < mass_max))
    subset2_count = np.sum((np.array(subset2_masses) > mass_min) & (np.array(subset2_masses) < mass_max))
    subset3_count = np.sum((np.array(subset3_masses) > mass_min) & (np.array(subset3_masses) < mass_max))
    subset4_count = np.sum((np.array(subset4_masses) > mass_min) & (np.array(subset4_masses) < mass_max))
    subset5_count = np.sum((np.array(subset5_masses) > mass_min) & (np.array(subset5_masses) < mass_max))

    # Compute counts for the Validation Set (with Training)
    total_count_trained = np.sum((np.array(all_pi0_masses_trained) > mass_min) & (np.array(all_pi0_masses_trained) < mass_max))
    subset2_count_trained = np.sum((np.array(subset2_masses_trained) > mass_min) & (np.array(subset2_masses_trained) < mass_max))
    subset3_count_trained = np.sum((np.array(subset3_masses_trained) > mass_min) & (np.array(subset3_masses_trained) < mass_max))
    subset4_count_trained = np.sum((np.array(subset4_masses_trained) > mass_min) & (np.array(subset4_masses_trained) < mass_max))
    subset5_count_trained = np.sum((np.array(subset5_masses_trained) > mass_min) & (np.array(subset5_masses_trained) < mass_max))

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # ---------------------------
    # Plot histograms for the Validation Set (OLD)
    # ---------------------------
    axs[0].hist(all_pi0_masses, range=xlimits, bins=bins, color="black", alpha=0.25)
    axs[0].hist(all_pi0_masses, range=xlimits, bins=bins, color="black", histtype='step', linewidth=2,
                label=f"Total (N={total_count})")

    axs[0].hist(subset2_masses, range=xlimits, bins=bins, color="red", alpha=0.25)
    axs[0].hist(subset2_masses, range=xlimits, bins=bins, color="red", histtype='step', linewidth=2,
                label=f"True Pi0's (N={subset2_count})")

    axs[0].hist(subset3_masses, range=xlimits, bins=bins, color="blue", alpha=0.25)
    axs[0].hist(subset3_masses, range=xlimits, bins=bins, color="blue", histtype='step', linewidth=2,
                label=f"Bkg Type A (N={subset3_count})")

    axs[0].hist(subset4_masses, range=xlimits, bins=bins, color="yellow", alpha=0.25)
    axs[0].hist(subset4_masses, range=xlimits, bins=bins, color="yellow", histtype='step', linewidth=2,
                label=f"Bkg Type B (N={subset4_count})")

    axs[0].hist(subset5_masses, range=xlimits, bins=bins, color="green", alpha=0.25)
    axs[0].hist(subset5_masses, range=xlimits, bins=bins, color="green", histtype='step', linewidth=2,
                label=f"Bkg Type C (N={subset5_count})")

    # ---------------------------
    # Plot histograms for the Validation Set (with Training) (NEW)
    # ---------------------------
    axs[1].hist(all_pi0_masses_trained, range=xlimits, bins=bins, color="black", alpha=0.25)
    axs[1].hist(all_pi0_masses_trained, range=xlimits, bins=bins, color="black", histtype='step', linewidth=2,
                label=f"Total (N={total_count_trained})")

    axs[1].hist(subset2_masses_trained, range=xlimits, bins=bins, color="red", alpha=0.25)
    axs[1].hist(subset2_masses_trained, range=xlimits, bins=bins, color="red", histtype='step', linewidth=2,
                label=f"True Pi0's (N={subset2_count_trained})")

    axs[1].hist(subset3_masses_trained, range=xlimits, bins=bins, color="blue", alpha=0.25)
    axs[1].hist(subset3_masses_trained, range=xlimits, bins=bins, color="blue", histtype='step', linewidth=2,
                label=f"Bkg Type A (N={subset3_count_trained})")

    axs[1].hist(subset4_masses_trained, range=xlimits, bins=bins, color="yellow", alpha=0.25)
    axs[1].hist(subset4_masses_trained, range=xlimits, bins=bins, color="yellow", histtype='step', linewidth=2,
                label=f"Bkg Type B (N={subset4_count_trained})")

    axs[1].hist(subset5_masses_trained, range=xlimits, bins=bins, color="green", alpha=0.25)
    axs[1].hist(subset5_masses_trained, range=xlimits, bins=bins, color="green", histtype='step', linewidth=2,
                label=f"Bkg Type C (N={subset5_count_trained})")

    # Modify axes limits and labels
    axs[0].set_xlim(xlimits)
    axs[0].set_xlabel("Diphoton Invariant Mass [GeV]")
    axs[1].set_xlabel("Diphoton Invariant Mass [GeV]")
    axs[0].set_ylabel("Counts")

    axs[0].set_title("Validation Set")
    axs[1].set_title("Validation Set (with Training)")

    axs[0].legend(fontsize=12)
    axs[1].legend(fontsize=12)

    plt.suptitle(f"N = Number of events in ${mass_min}<M<{mass_max}$ GeV",fontsize=20)
    plt.tight_layout()
    # Save figure
    filename = os.path.join(output_dir, 'pi0_invariant_mass.png')
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