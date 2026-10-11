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
 
 
# --------------------------------------------------------------------------
# 2. MISSING VALUE ANALYSIS
# --------------------------------------------------------------------------
def missing_analysis(df, out):
    miss = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    }).sort_values("missing_count", ascending=False)
    miss.to_csv(f"{out}/missing_values.csv")
 
    shown = miss[miss.missing_count > 0]
    if len(shown):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(shown) + 1)))
        sns.barplot(x=shown.missing_pct, y=shown.index, ax=ax, color="#d9534f")
        ax.set_xlabel("% missing")
        ax.set_title("Missing Values by Column")
        for i, v in enumerate(shown.missing_pct):
            ax.text(v + 0.05, i, f"{v}%", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{out}/01_missing_values.png", dpi=150)
        plt.close(fig)
    return miss
 
 
# --------------------------------------------------------------------------
# 3. CLEANING
# --------------------------------------------------------------------------
def clean_data(df, grade_cols):
    log = []
    df = df.copy()
 
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"Removed {n0 - len(df)} duplicate rows.")
 
    # Invalid ranges -> NaN (then imputed)
    if "age" in df.columns:
        bad = ((df.age < 10) | (df.age > 30)).sum()
        df.loc[(df.age < 10) | (df.age > 30), "age"] = np.nan
        log.append(f"Flagged {bad} implausible age values (outside 10-30) as missing.")
    for g in grade_cols:
        bad = ((df[g] < 0) | (df[g] > 20)).sum()
        df.loc[(df[g] < 0) | (df[g] > 20), g] = np.nan
        if bad:
            log.append(f"Flagged {bad} out-of-range values in {g} (valid range 0-20) as missing.")
 
    # Impute: median for numeric, mode for categorical
    for col in df.columns:
        n_na = df[col].isna().sum()
        if n_na == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            log.append(f"Imputed {n_na} missing '{col}' values with the median.")
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            log.append(f"Imputed {n_na} missing '{col}' values with the mode.")
 
    # Final grade-based labels
    return df, log
 
 
# --------------------------------------------------------------------------
# 4. STATISTICAL SUMMARY
# --------------------------------------------------------------------------
def stat_summary(df, out):
    num = df.select_dtypes("number")
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["median"] = num.median()
    desc.round(2).to_csv(f"{out}/statistical_summary.csv")
    cat = df.select_dtypes(exclude="number")
    if not cat.empty:
        cat.describe().T.to_csv(f"{out}/categorical_summary.csv")
    return desc
 
 
# --------------------------------------------------------------------------
# 5-7. CHARTS
# --------------------------------------------------------------------------
def histograms(df, target, out):
    cols = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 2]
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows))
    for ax, c in zip(np.ravel(axes), cols):
        sns.histplot(df[c], bins=min(20, df[c].nunique()), kde=df[c].nunique() > 8,
                     ax=ax, color="#4c72b0")
        ax.set_title(c)
        ax.set_xlabel("")
    for ax in np.ravel(axes)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distributions of Numeric Variables", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{out}/02_histograms.png", dpi=150)
    plt.close(fig)
 
    # Dedicated final-grade histogram with mean/median lines
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[target], bins=21, kde=True, ax=ax, color="#55a868")
    ax.axvline(df[target].mean(), color="red", ls="--", label=f"Mean {df[target].mean():.1f}")
    ax.axvline(df[target].median(), color="black", ls=":", label=f"Median {df[target].median():.1f}")
    ax.set_title(f"Distribution of Final Grade ({target})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/03_final_grade_hist.png", dpi=150)
    plt.close(fig)
 
 
def boxplots(df, target, out):
    # (a) target grouped by key categorical / ordinal factors
    candidates = [c for c in ["studytime", "failures", "sex", "internet", "address",
                              "school", "famsup", "Medu", "Dalc"] if c in df.columns]
    candidates = candidates[:6]
    ncols = 3
    nrows = int(np.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    for ax, c in zip(np.ravel(axes), candidates):
        sns.boxplot(data=df, x=c, y=target, ax=ax, hue=c, legend=False, palette="Set2")
        ax.set_title(f"{target} by {c}")
    for ax in np.ravel(axes)[len(candidates):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{out}/04_boxplots_by_factor.png", dpi=150)
    plt.close(fig)
 
    # (b) outlier view across grade columns & absences
    grade_cols = [c for c in ["G1", "G2", "G3"] if c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df[grade_cols], ax=axes[0], palette="Set2")
    axes[0].set_title("Grade Progression (G1 -> G2 -> G3)")
    if "absences" in df.columns:
        sns.boxplot(x=df["absences"], ax=axes[1], color="#c44e52")
        axes[1].set_title("Absences (outlier check)")
    fig.tight_layout()
    fig.savefig(f"{out}/05_boxplots_outliers.png", dpi=150)
    plt.close(fig)
 
 
def correlation_heatmap(df, target, out):
    num = df.select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(f"{out}/06_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    return corr
 
 
# --------------------------------------------------------------------------
# 8. INSIGHTS REPORT
# --------------------------------------------------------------------------
def build_report(df, raw_shape, miss, log, desc, corr, target, out):
    L = []
    add = L.append
    t = df[target]
    pass_mark = 10
    pass_rate = (t >= pass_mark).mean() * 100
 
    add("# Student Performance — Insights Report\n")
    add("## 1. Dataset Overview")
    add(f"- Raw data: **{raw_shape[0]} rows × {raw_shape[1]} columns**; after cleaning: **{len(df)} rows**.")
    add(f"- Target variable: `{target}` (final grade, 0–20 scale).\n")
 
    add("## 2. Data Quality & Cleaning")
    top = miss[miss.missing_count > 0].head(5)
    if len(top):
        add("Columns with the most missing values: " +
            ", ".join(f"`{i}` ({r.missing_pct}%)" for i, r in top.iterrows()) + ".\n")
    for item in log:
        add(f"- {item}")
    add("")
 
    add("## 3. Statistical Summary")
    add(f"- Mean final grade **{t.mean():.2f}**, median **{t.median():.1f}**, std **{t.std():.2f}**, "
        f"range {t.min():.0f}–{t.max():.0f}.")
    add(f"- **{pass_rate:.1f}%** of students scored ≥ {pass_mark} (pass); "
        f"**{(t == 0).mean() * 100:.1f}%** scored 0 (possible drop-outs / non-attendance).")
    add(f"- Skewness of `{target}`: {t.skew():.2f} "
        f"({'left-skewed' if t.skew() < -0.3 else 'right-skewed' if t.skew() > 0.3 else 'roughly symmetric'}).\n")
    add("![Final grade](03_final_grade_hist.png)\n")
    add("![Histograms](02_histograms.png)\n")
 
    add("## 4. Key Findings")
    # correlations with target
    ct = corr[target].drop(target).sort_values(key=abs, ascending=False)
    add("**Strongest correlations with final grade:**\n")
    for k, v in ct.head(5).items():
        direction = "positive" if v > 0 else "negative"
        add(f"- `{k}`: r = {v:+.2f} ({direction})")
    add("")
    if {"G1", "G2"} <= set(df.columns):
        add(f"- **Prior performance dominates.** G1 and G2 correlate {corr.loc['G1', target]:.2f} and "
            f"{corr.loc['G2', target]:.2f} with final grade, so early-term grades are the best "
            f"early-warning signal.")
    if "failures" in df.columns:
        g = df.groupby("failures")[target].mean().round(2)
        add(f"- **Past failures hurt.** Mean final grade by number of past failures: "
            + ", ".join(f"{k} → {v}" for k, v in g.items()) + ".")
    if "studytime" in df.columns:
        g = df.groupby("studytime")[target].mean().round(2)
        add(f"- **Study time helps.** Mean final grade by study-time band (1=low … 4=high): "
            + ", ".join(f"{k} → {v}" for k, v in g.items()) + ".")
    if "absences" in df.columns:
        hi = df[df.absences > df.absences.quantile(.75)][target].mean()
        lo = df[df.absences <= df.absences.quantile(.25)][target].mean()
        add(f"- **Absences:** students in the top quartile of absences average {hi:.2f} vs "
            f"{lo:.2f} for the bottom quartile (r = {corr.loc['absences', target]:+.2f}).")
    for cat in ["sex", "internet", "address", "school"]:
        if cat in df.columns:
            g = df.groupby(cat)[target].mean().round(2)
            gap = g.max() - g.min()
            add(f"- `{cat}`: " + ", ".join(f"{k} = {v}" for k, v in g.items()) +
                f" (gap {gap:.2f} points{' — small' if gap < 1 else ''}).")
    if {"G1", "G3"} <= set(df.columns):
        d = (df["G3"] - df["G1"])
        add(f"- **Grade trajectory:** average change from G1 to G3 is {d.mean():+.2f}; "
            f"{(d < -2).mean() * 100:.1f}% of students dropped by more than 2 points and "
            f"{(d > 2).mean() * 100:.1f}% improved by more than 2 points.")
    add("")
    add("![Boxplots](04_boxplots_by_factor.png)\n")
    add("![Outliers](05_boxplots_outliers.png)\n")
    add("![Correlation heatmap](06_correlation_heatmap.png)\n")
 
    add("## 5. Recommendations")
    add("1. **Early-warning system:** flag students with low G1/G2 or prior failures for tutoring before the final term.")
    add("2. **Attendance monitoring:** intervene when absences enter the top quartile.")
    add("3. **Study-skills support:** promote structured study time for students in the lowest study-time band.")
    add("4. **Investigate zero-grade records:** determine whether they are drop-outs, missing exams or data-entry errors.")
    add("5. **Equity checks:** review gaps by school, address and internet access for resource inequality.\n")
 
    add("## 6. Limitations")
    add("- Correlation is not causation; observational data cannot prove that study time *causes* higher grades.")
    add("- Median/mode imputation reduces variance; results may differ with multiple imputation.")
    add("- Findings from a single cohort may not generalise to other schools or years.")
 
    with open(f"{out}/insights_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)
 
 
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Student Performance Data Explorer")
    ap.add_argument("--data", default="student-mat.csv", help="Path to CSV")
    ap.add_argument("--target", default="G3", help="Final grade column (default G3)")
    ap.add_argument("--out", default="explorer_output", help="Output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
 
    df_raw = load_data(args.data)
    if args.target not in df_raw.columns:
        raise SystemExit(f"Target column '{args.target}' not found. Columns: {list(df_raw.columns)}")
 
    print("[missing] analysing...")
    miss = missing_analysis(df_raw, args.out)
 
    print("[clean] cleaning...")
    grade_cols = [c for c in ["G1", "G2", "G3"] if c in df_raw.columns]
    df, log = clean_data(df_raw, grade_cols)
    df.to_csv(f"{args.out}/cleaned_data.csv", index=False)
 
    print("[stats] summarising...")
    desc = stat_summary(df, args.out)
 
    print("[charts] drawing...")
    histograms(df, args.target, args.out)
    boxplots(df, args.target, args.out)
    corr = correlation_heatmap(df, args.target, args.out)
 
    print("[report] writing insights...")
    build_report(df, df_raw.shape, miss, log, desc, corr, args.target, args.out)
    print(f"\nDone. Everything saved in ./{args.out}/")
 
 
if __name__ == "__main__":
    main()