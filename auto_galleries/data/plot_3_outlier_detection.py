"""
========================================
Outlier Detection
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


# %%
# Outlier detection by CBLOF
results = ds.detect_outlier_cblof(dataset="main", method="kmeans", threshold=0.9)
results.plot()

# %%
# Outlier detection by Isolation forest
results = ds.detect_outlier_isolation_forest()
results.plot()

# %%
# Outlier detection by PCA
results = ds.detect_outlier_pca(dataset="main", method="reconst_error")
outliers_sample_index = results.table['outliers'].index
results.plot()

# %%
# Outliers table
results.table['outliers']

# %%
# non-outliers table
results.table['non-outliers']

# %%
# Evaluate outlier scores of samples
results.func(results.table['outliers'])

# %%
# Evaluate outlier scores of samples
results.func(results.table['non-outliers'])

# %%
# Apply outlier removal
ds.set_inactive_samples(dataset="main", sample_idx=outliers_sample_index)
ds.x.shape
