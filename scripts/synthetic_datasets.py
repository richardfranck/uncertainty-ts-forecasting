"""
This generates the synthetic tumor data and is based on the class defined in TIMEVIEW .
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint


class SyntheticTumorDataset:
    """Generate synthetic tumor trajectories under chemotherapy.

    This dataset produces:
      - X: pandas DataFrame with static covariates
      - ts: list of numpy arrays with time grids
      - ys: list of numpy arrays with tumor volume trajectories
    """

    def __init__(
        self,
        n_samples=2000,
        n_time_steps=20,
        time_horizon=1.0,
        noise_std=0.0,
        seed=0,
        equation="wilkerson",
    ):
        """Initialize and generate a synthetic tumor dataset.

        Args:
            n_samples: Number of simulated patients.
            n_time_steps: Number of time points per trajectory.
            time_horizon: Final time value of the simulation (times range [0, time_horizon]).
            noise_std: Standard deviation of Gaussian observation noise.
            seed: Random seed used for sampling covariates and noise.
            equation: Tumor model to use. Supported values: "wilkerson", "geng".

        Raises:
            ValueError: If equation is not one of the supported values.
        """
        if equation not in ["wilkerson", "geng"]:
            raise ValueError('equation must be one of: "wilkerson", "geng".')

        self.n_samples = n_samples
        self.n_time_steps = n_time_steps
        self.time_horizon = time_horizon
        self.noise_std = noise_std
        self.seed = seed
        self.equation = equation

        X, ts, ys = self.synthetic_tumor_data(
            n_samples=self.n_samples,
            n_time_steps=self.n_time_steps,
            time_horizon=self.time_horizon,
            noise_std=self.noise_std,
            seed=self.seed,
            equation=self.equation,
        )

        if self.equation == "wilkerson":
            self.X = pd.DataFrame(X, columns=["age", "weight", "initial_tumor_volume", "dosage"])
        elif self.equation == "geng":
            self.X = pd.DataFrame(X, columns=["age", "weight", "initial_tumor_volume", "start_time", "dosage"])

        self.ts = ts
        self.ys = ys

    def get_X_ts_ys(self):
        """Return the generated dataset.

        Returns:
            X: pandas DataFrame of static covariates.
            ts: list of numpy arrays containing time grids.
            ys: list of numpy arrays containing tumor trajectories.
        """
        return self.X, self.ts, self.ys

    def __len__(self):
        """Return number of samples."""
        return len(self.X)

    def get_feature_ranges(self):
        """Return the ranges used for sampling each feature.

        Returns:
            Dictionary mapping each feature name to (min, max).
        """
        if self.equation == "wilkerson":
            return {
                "age": (20, 80),
                "weight": (40, 100),
                "initial_tumor_volume": (0.1, 0.5),
                "dosage": (0.0, 1.0),
            }
        elif self.equation == "geng":
            return {
                "age": (20, 80),
                "weight": (40, 100),
                "initial_tumor_volume": (0.1, 0.5),
                "start_time": (0.0, 1.0),
                "dosage": (0.0, 1.0),
            }

    def get_feature_names(self):
        """Return the names of the static covariates.

        Returns:
            List of feature name strings.
        """
        if self.equation == "wilkerson":
            return ["age", "weight", "initial_tumor_volume", "dosage"]
        elif self.equation == "geng":
            return ["age", "weight", "initial_tumor_volume", "start_time", "dosage"]

    def _tumor_volume(self, t, age, weight, initial_tumor_volume, start_time, dosage):
        """Compute tumor volume under chemotherapy using an ODE-based model.

        This corresponds to the "geng" branch in the original code.

        Args:
            t: 1D numpy array of times at which to compute tumor volume.
            age: Age feature value.
            weight: Weight feature value.
            initial_tumor_volume: Initial tumor volume feature value.
            start_time: Start time of chemotherapy.
            dosage: Chemotherapy dosage.

        Returns:
            1D numpy array of tumor volumes aligned with t.
        """
        RHO_0 = 2.0

        K_0 = 1.0
        K_1 = 0.01

        BETA_0 = 50.0

        GAMMA_0 = 5.0

        V_min = 0.001

        rho = RHO_0 * (age / 20.0) ** 0.5
        K = K_0 + K_1 * (weight)
        beta = BETA_0 * (age / 20.0) ** (-0.2)

        def C(t_scalar):
            return np.where(
                t_scalar < start_time,
                0.0,
                dosage * np.exp(-GAMMA_0 * (t_scalar - start_time)),
            )

        def dVdt(V, t_scalar):
            return rho * (V - V_min) * V * np.log(K / V) - beta * V * C(t_scalar)

        V = odeint(dVdt, initial_tumor_volume, t)[:, 0]
        return V

    def _tumor_volume_2(self, t, age, weight, initial_tumor_volume, dosage):
        """Compute tumor volume using a closed-form model.

        This corresponds to the "wilkerson" branch in the original code.

        Args:
            t: 1D numpy array of times at which to compute tumor volume.
            age: Age feature value.
            weight: Weight feature value.
            initial_tumor_volume: Initial tumor volume feature value.
            dosage: Chemotherapy dosage.

        Returns:
            1D numpy array of tumor volumes aligned with t.
        """
        G_0 = 2.0
        D_0 = 180.0
        PHI_0 = 10.0

        g = G_0 * (age / 20.0) ** 0.5
        d = D_0 * dosage / weight
        phi = 1.0 / (1.0 + np.exp(-dosage * PHI_0))

        return initial_tumor_volume * (phi * np.exp(-d * t) + (1.0 - phi) * np.exp(g * t))

    def synthetic_tumor_data(self, n_samples, n_time_steps, time_horizon=1.0, noise_std=0.0, seed=0, equation="wilkerson"):
        """Generate synthetic tumor data based on a chemotherapy tumor model.

        Executes:
          1) sample static covariates uniformly from fixed ranges
          2) create a time grid per sample
          3) simulate tumor volume trajectories using the chosen equation
          4) add Gaussian observation noise

        Args:
            n_samples: Number of samples to generate.
            n_time_steps: Number of time points per trajectory.
            time_horizon: Final time value (times range [0, time_horizon]).
            noise_std: Standard deviation of Gaussian observation noise.
            seed: Random seed.
            equation: "wilkerson" or "geng".

        Returns:
            X: numpy array of static covariates with shape (n_samples, 4 or 5).
            ts: list of numpy arrays, each of length n_time_steps.
            ys: list of numpy arrays, each of length n_time_steps.
        """
        feature_ranges = {
            "age": (20, 80),
            "weight": (40, 100),
            "initial_tumor_volume": (0.1, 0.5),
            "start_time": (0.0, 1.0),
            "dosage": (0.0, 1.0),
        }

        gen = np.random.default_rng(seed)

        age = gen.uniform(feature_ranges["age"][0], feature_ranges["age"][1], size=n_samples)
        weight = gen.uniform(feature_ranges["weight"][0], feature_ranges["weight"][1], size=n_samples)
        tumor_volume = gen.uniform(
            feature_ranges["initial_tumor_volume"][0],
            feature_ranges["initial_tumor_volume"][1],
            size=n_samples,
        )
        start_time = gen.uniform(feature_ranges["start_time"][0], feature_ranges["start_time"][1], size=n_samples)
        dosage = gen.uniform(feature_ranges["dosage"][0], feature_ranges["dosage"][1], size=n_samples)

        if equation == "wilkerson":
            X = np.stack((age, weight, tumor_volume, dosage), axis=1)
        elif equation == "geng":
            X = np.stack((age, weight, tumor_volume, start_time, dosage), axis=1)

        ts = [np.linspace(0.0, time_horizon, n_time_steps) for _ in range(n_samples)]
        ys = []

        for i in range(n_samples):
            if equation == "wilkerson":
                age_i, weight_i, vol0_i, dose_i = X[i, :]
                y = self._tumor_volume_2(ts[i], age_i, weight_i, vol0_i, dose_i)
            elif equation == "geng":
                age_i, weight_i, vol0_i, start_i, dose_i = X[i, :]
                y = self._tumor_volume(ts[i], age_i, weight_i, vol0_i, start_i, dose_i)

            y = y + gen.normal(0.0, noise_std, size=n_time_steps)
            ys.append(y)

        return X, ts, ys

    def get_feature_types(self):
        """Return feature types for interactive visualization.

        The TIMEVIEW visualization code expects each feature to have a type so it
        knows how to build the UI controls (e.g., slider vs dropdown).

        Returns:
            Dictionary mapping feature name -> feature type string.
            For this synthetic dataset, all features are continuous.
        """
        types = {}
        for name in self.get_feature_names():
            types[name] = "continuous"
        return types



if __name__ == "__main__":
    """ Inpect some sample data. """
    import matplotlib.pyplot as plt

    # Create a small dataset so the plot is readable
    ds = SyntheticTumorDataset(
        n_samples=10,
        n_time_steps=50,
        time_horizon=1.0,
        noise_std=0.05,
        seed=0,
        equation="wilkerson",
    )

    print("Feature names:", ds.get_feature_names())
    print("Feature ranges:", ds.get_feature_ranges())
    print("Feature types:", ds.get_feature_types())

    X, ts, ys = ds.get_X_ts_ys()

    print("X (static covariates DataFrame)")
    print(X)

    print("DataFrame shape (rows, columns)")
    print(X.shape)

    print("First row (patient 0)")
    print(X.iloc[0])
  

    # Plot the figure for the first patient. 
    plt.figure()
    plt.plot(ts[0], ys[0])
    plt.xlabel("time")
    plt.ylabel("tumor volume")
    plt.title("Synthetic tumor trajectory (first individual in panel data)")
    plt.show()



