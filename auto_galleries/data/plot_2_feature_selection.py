"""
========================================
Feature Selection
========================================

"""

# %%
# Import modeva modules
import numpy as np
import pandas as pd
from modeva import DataSet

# %%
# Load a simulated Friedman data
from sklearn.datasets import make_friedman1
x, y = make_friedman1(n_samples=10000, n_features=10, noise=0.1, random_state=2024)
columns = ['X' + str(i) for i in range(10)] + ['Y']
df = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], 1), columns=columns)

ds = DataSet()
ds.load_dataframe(data=df)
ds.set_random_split()

# %%
# Correlation based feature selection
results = ds.feature_select_corr(threshold=0.2)
results.plot()

# %%
# XGB-PFI based feature selection
results = ds.feature_select_xgbpfi(threshold=0.01)
results.plot()

# %%
# RCIT based feature selection
results = ds.feature_select_rcit()
results.plot()

# %%
# Apply feature selection
ds.set_active_features(features=results.value["selected"])
ds.feature_names

# %%
# Conduct another round of feature selection
results = ds.feature_select_xgbpfi(threshold=0.1)
results.plot()

# %%
# Apply another round of feature selection
ds.set_active_features(features=results.value["selected"])
ds.feature_names

# %%
# Revert all feature selection
ds.set_active_features(features=None) # by default, all features are set active
ds.feature_names
