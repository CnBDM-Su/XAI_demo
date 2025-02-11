"""
========================================
Subsampling
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
x, y = make_friedman1(n_samples=100000, n_features=10, noise=0.1, random_state=0)
columns = ['X' + str(i) for i in range(10)] + ['target']
df = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], 1), columns=columns)

ds = DataSet(name="subsampling-demo")
ds.load_dataframe(df)

# %%
# Random subsampling
ds.set_active_samples()
results = ds.subsample_random(dataset="main", sample_size=1000)
active_samples_index = results.value["sample_idx"]
active_samples_index

# %%
# Apply subsampling by setting active samples
ds.set_active_samples(dataset="main", sample_idx=active_samples_index)
ds.x.shape

# %%
# Reset subsampling by `ds.set_active_samples()`
ds.set_active_samples(dataset="main", sample_idx=None)
ds.x.shape
