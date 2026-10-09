"""
[Student Performance Data Explorer]

Pipeline: load -> missing-value analysis -> cleaning -> statistical summary
          -> histograms -> box plots -> correlation heatmap -> insights report
 
Usage:
    python student_explorer.py                      # uses built-in synthetic demo data
    python student_explorer.py --data student-mat.csv   # your own CSV (comma or ';' separated)
    python student_explorer.py --data data.csv --target G3 --out results
 
Requires: pandas, numpy, matplotlib, seaborn
"""

import argparse
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------
def make_demo_data(n=500, seed=42):
    """Synthetic data shaped like the UCI 'Student Performance' dataset,
    with deliberately injected missing values, duplicates and bad entries."""
    rng = np.random.default_rng(seed)
    studytime = rng.choice([1, 2, 3, 4], n, p=[0.27, 0.50, 0.16, 0.07])
    failures = rng.choice([0, 1, 2, 3], n, p=[0.78, 0.13, 0.05, 0.04])
    absences = np.clip(rng.poisson(5, n) + (rng.random(n) < 0.04) * rng.integers(20, 60, n), 0, 75)
    medu = rng.integers(0, 5, n)
    fedu = np.clip(medu + rng.integers(-1, 2, n), 0, 4)
    goout = rng.integers(1, 6, n)
    dalc = rng.choice([1, 2, 3, 4, 5], n, p=[0.70, 0.19, 0.06, 0.03, 0.02])
    health = rng.integers(1, 6, n)
    age = rng.choice([15, 16, 17, 18, 19, 20], n, p=[0.2, 0.28, 0.26, 0.2, 0.05, 0.01])
 
    ability = rng.normal(11, 2.5, n)
    g1 = ability + 0.6 * studytime - 1.2 * failures + 0.3 * medu - 0.3 * dalc + rng.normal(0, 1.5, n)
    g2 = g1 + rng.normal(0.2, 1.2, n) - 0.04 * absences
    g3 = g2 + rng.normal(0.1, 1.3, n) - 0.03 * absences
    clip = lambda x: np.clip(np.round(x), 0, 20).astype(int)
 
    df = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], n, p=[0.88, 0.12]),
        "sex": rng.choice(["F", "M"], n),
        "age": age,
        "address": rng.choice(["U", "R"], n, p=[0.78, 0.22]),
        "Medu": medu, "Fedu": fedu,
        "studytime": studytime, "failures": failures,
        "internet": rng.choice(["yes", "no"], n, p=[0.83, 0.17]),
        "famsup": rng.choice(["yes", "no"], n, p=[0.62, 0.38]),
        "goout": goout, "Dalc": dalc, "health": health,
        "absences": absences,
        "G1": clip(g1), "G2": clip(g2), "G3": clip(g3),
    })
    # --- inject data-quality problems ---
    for col, frac in [("absences", .04), ("studytime", .03), ("Medu", .05),
                      ("internet", .04), ("G2", .02), ("health", .03)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "age"] = 150          # impossible values
    df.loc[rng.choice(n, 2, replace=False), "G1"] = 45            # out of 0-20 range
    df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)  # duplicates
    return df

def load_data(path):
    if path is None:
        print("[load] No --data supplied; using synthetic demo dataset.")
        return make_demo_data()
    df = pd.read_csv(path, sep=None, engine="python")  # auto-detects , or ;
    print(f"[load] Loaded {path}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df

