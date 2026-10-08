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
