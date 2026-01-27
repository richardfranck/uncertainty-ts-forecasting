"""
Bootstrap Epistemic Uncertainty Analysis with Fixed Knots

This trains an ensemble of TIMEVIEW models using bootstrap resampling. All models share
fixed internal knots.
"""

import warnings
import re
warnings.showwarning = lambda m, *a, **k: print(re.sub(r"/Users/[^/]+/", "/Users/USER/", str(m)))
from pathlib import Path
import json
import numpy as np
import pandas as pd
from timeview.config import Config
from timeview.data import TTSDataset
from timeview.lit_module import LitTTS
from timeview.training import training
from timeview.knot_selection import calculate_knot_placement  
from synthetic_datasets import SyntheticTumorDataset
import pickle  

# Data generation parameters
DATA_SEED = 0
N_SAMPLES = 2000
N_TIME_STEPS = 20
TIME_HORIZON = 1.0
NOISE_STD = 0.0
EQUATION = "wilkerson"

# Model architecture parameters
N_BASIS = 9
DEVICE = "cpu"

# Bootstrap parameters
MAX_BOOTSTRAPS = 30
BOOTSTRAP_SEED = 12345 # Bootstrap seed controls which patients are sampled (with replacement) to create each bootstrap dataset.
TRAIN_SEED = 0 # Train seed is fixed at 0, so all models start from the same neural network initialization

# Analysis parameters
T_GRID_POINTS = 200

# Output directory
OUTDIR = Path("outputs_uncertainty")
OUTDIR.mkdir(parents=True, exist_ok=True)



def generate_synthetic_dataset(n_samples, n_time_steps, time_horizon, 
                               noise_std, seed, equation):
    """Generate synthetic tumor growth dataset.
    
    Args:
        n_samples: Number of patient samples to generate.
        n_time_steps: Number of time points per trajectory.
        time_horizon: Maximum time value (trajectories span [0, time_horizon]).
        noise_std: Standard deviation of Gaussian observation noise.
        seed: Random seed for reproducibility.
        equation: Tumor growth model to use ('wilkerson' or 'geng').
    
    Returns:
        Tuple of (X, ts, ys) where:
            X: pandas DataFrame of static patient covariates (n_samples x n_features)
            ts: list of numpy arrays containing time points for each patient
            ys: list of numpy arrays containing tumor volumes for each patient
    """
    dataset = SyntheticTumorDataset(
        n_samples=n_samples,
        n_time_steps=n_time_steps,
        time_horizon=time_horizon,
        noise_std=noise_std,
        seed=seed,
        equation=equation,
    )
    return dataset.get_X_ts_ys()


def compute_fixed_knots(ts, ys, n_basis, time_horizon, seed, verbose=True):
    """Calculate fixed internal knots from the full dataset.
    
    Internal knots define the positions where B-spline polynomial pieces
    connect. Using fixed knots across all bootstrap models ensures that
    trajectory compositions are directly comparable.
    
    Args:
        ts: List of time point arrays for all patients.
        ys: List of tumor volume arrays for all patients.
        n_basis: Number of B-spline basis functions.
        time_horizon: Maximum time value.
        seed: Random seed for K-means clustering in knot placement.
        verbose: Whether to print progress information.
    
    Returns:
        numpy array of internal knot positions.
    """
    n_internal_knots = n_basis - 2  # For cubic B-splines (degree k=3)
    
    internal_knots = calculate_knot_placement(
        ts, ys,
        n_internal_knots=n_internal_knots,
        T=time_horizon,
        seed=seed,
        verbose=verbose
    )
    
    return internal_knots


def create_bootstrap_sample(X, ts, ys, n_patients, rng):
    """Create a bootstrap resample of the dataset.
    
    Samples patient indices with replacement to create a new dataset
    of the same size as the original.
    
    Args:
        X: pandas DataFrame of static covariates.
        ts: List of time point arrays.
        ys: List of tumor volume arrays.
        n_patients: Number of patients in original dataset.
        rng: numpy random number generator.
    
    Returns:
        Tuple of (X_boot, ts_boot, ys_boot) representing the bootstrap sample.
    """
    # Sample patient indices with replacement
    bootstrap_indices = rng.integers(low=0, high=n_patients, size=n_patients)
    
    # Create bootstrap dataset
    X_boot = X.iloc[bootstrap_indices].reset_index(drop=True)
    ts_boot = [ts[i] for i in bootstrap_indices]
    ys_boot = [ys[i] for i in bootstrap_indices]
    
    return X_boot, ts_boot, ys_boot


def train_bootstrap_ensemble(X, ts, ys, config, n_bootstraps, bootstrap_seed, 
                             train_seed):
    """Train an ensemble of models using bootstrap resampling.
    
    Args:
        X: pandas DataFrame of static covariates.
        ts: List of time point arrays.
        ys: List of tumor volume arrays.
        config: TIMEVIEW Config object with fixed internal knots.
        n_bootstraps: Number of bootstrap replicates to train.
        bootstrap_seed: Seed for bootstrap sampling randomness.
        train_seed: Seed for neural network training.
    
    Returns:
        List of trained LitTTS model objects.
    """
    n_patients = len(ts)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_models = []
    
    for b in range(n_bootstraps):
        print("")
        print("=" * 60)
        print(f"Bootstrap replicate {b + 1}/{n_bootstraps}")
        print("=" * 60)
        
        # Create bootstrap sample
        X_boot, ts_boot, ys_boot = create_bootstrap_sample(
            X, ts, ys, n_patients, rng
        )
        
        # Create TIMEVIEW dataset
        dataset_boot = TTSDataset(config, (X_boot, ts_boot, ys_boot))
        
        # Train model
        model = training(
            seed=train_seed,
            config=config,
            dataset=dataset_boot
        )
        
        bootstrap_models.append(model)
    
    return bootstrap_models


def save_results(outdir, bootstrap_models, config, dataset, metadata):
    """Save all results to disk.
    
    Args:
        outdir: Path object for output directory.
        bootstrap_models: List of trained models.
        config: TIMEVIEW Config object.
        dataset: Tuple of (X, ts, ys).
        metadata: Dictionary of run metadata.
    """
    X, ts, ys = dataset
    
    # Save bootstrap models
    models_path = outdir / "bootstrap_models.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(bootstrap_models, f)
    print(f"Saved models to: {models_path}")
    
    # Save config
    config_path = outdir / "config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"Saved config to: {config_path}")
    
    # Save original dataset
    dataset_path = outdir / "dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump((X, ts, ys), f)
    print(f"Saved dataset to: {dataset_path}")
    
    # Save metadata
    meta_path = outdir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_path}")




def main():
    # Step 1: Generate synthetic dataset
    X, ts, ys = generate_synthetic_dataset(
        n_samples=N_SAMPLES,
        n_time_steps=N_TIME_STEPS,
        time_horizon=TIME_HORIZON,
        noise_std=NOISE_STD,
        seed=DATA_SEED,
        equation=EQUATION
    )
    
    # Step 2: Calculate fixed internal knots
    internal_knots = compute_fixed_knots(
        ts, ys,
        n_basis=N_BASIS,
        time_horizon=TIME_HORIZON,
        seed=DATA_SEED,
        verbose=True
    )
  
    # Step 3: Create TIMEVIEW configuration
    config = Config(
        n_features=X.shape[1],
        n_basis=N_BASIS,
        T=TIME_HORIZON,
        seed=0,
        device=DEVICE,
        internal_knots=internal_knots
    )
    
    # Step 4: Train bootstrap ensemble
    bootstrap_models = train_bootstrap_ensemble(
        X, ts, ys,
        config=config,
        n_bootstraps=MAX_BOOTSTRAPS,
        bootstrap_seed=BOOTSTRAP_SEED,
        train_seed=TRAIN_SEED
    )
    
    # Step 5: Save results
    metadata = {
        "method": "bootstrap",
        "data_seed": DATA_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "train_seed": TRAIN_SEED,
        "n_bootstraps": MAX_BOOTSTRAPS,
        "n_samples": N_SAMPLES,
        "n_time_steps": N_TIME_STEPS,
        "time_horizon": TIME_HORIZON,
        "noise_std": NOISE_STD,
        "equation": EQUATION,
        "n_basis": N_BASIS,
        "device": DEVICE,
        "fixed_internal_knots": [float(k) for k in internal_knots]
    }
    
    save_results(
        OUTDIR,
        bootstrap_models,
        config,
        (X, ts, ys),
        metadata
    )

if __name__ == "__main__":
    main()