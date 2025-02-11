"""
==============================================
Advance Usage: Data with Model Predictions
==============================================

"""

# %%
# Import modeva modules
import numpy as np
import pandas as pd
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoXGBRegressor
from modeva.models import MoScoredRegressor

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
# Fit a XGB model
model = MoXGBRegressor(max_depth=2)
model.fit(ds.train_x, ds.train_y)

# %%
# Get XGB predictions and combine it to original dataframe
data = ds.to_df()
data["prediction"] = model.predict(ds.x)
data

# %%
# Next, we will use this combined data to do model validation
new_ds = DataSet(name="scored-test-demo")
new_ds.load_dataframe(data)
new_ds.set_train_idx(train_idx=np.array(ds.train_idx))
new_ds.set_test_idx(test_idx=np.array(ds.test_idx))
new_ds.set_target(feature="Y")
new_ds.set_prediction(feature="prediction")
new_ds.register(override=True)

# %%
# Reload the model (optional)
reload_ds = DataSet(name="scored-test-demo")
reload_ds.load_registered_data(name="scored-test-demo")

# %%
# Run tests without the model object, note that the robustness test is not available for scored model
model = MoScoredRegressor(dataset=new_ds)
ts = TestSuite(ds, model)

# %%
# Run accuracy test without the model object
results = ts.diagnose_accuracy_table()
results.table

# %%
# Run residual analysis test without the model object
results = ts.diagnose_residual_analysis(features="X1")
results.table

# %%
# Run reliability test without the model object
results = ts.diagnose_reliability()
results.table

# %%
# Run resilience test without the model object
results = ts.diagnose_resilience()
results.table

# %%
# Run slicing accuracy test without the model object
results = ts.diagnose_slicing_accuracy(features="X1", dataset="main", metric="MAE", threshold=0)
results.table

# %%
# Run slicing overfit test without the model object
results = ts.diagnose_slicing_overfit(features="X1", train_dataset="train", test_dataset="test", metric="MAE")
results.table
