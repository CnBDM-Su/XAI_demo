"""
========================================
Basic Dataset Operations
========================================

"""
import pandas as pd
# %%
# Import modeva modules
from modeva import DataSet
from modeva.data.utils.loading import load_builtin_data

# %%
# Load a dataset in pandas DataFrame format
data = load_builtin_data("TaiwanCredit")
data

# %%
# Manually create some special values and missing values for demostration purpose
data["LIMIT_BAL"].iloc[:10] = "SV1"
data["PAY_1"].iloc[10:15] = "SV2"
data["EDUCATION"].iloc[5:20] = pd.NA
data["AGE"].iloc[0:5] = pd.NA
data

# %%
# Load the dataframe into Modeva
ds = DataSet(name="TW-Credit")
ds.load_dataframe(data)
ds.data.head(20).iloc[:, :10]

# %%
# Check if the data has missing values
results = ds.summary()
results.table["summary"]

# %%
# Check the features with special values.
# (mixed means that feature is a mixture of numerical and categorical)
results.table["mixed"]

# %%
# Preprocess the data
# ----------------------------
# Reset preprocessing
ds.reset_preprocess()

# %% Data imputation
# Impute numerical features, and add an indicator column for missing values
ds.impute_missing(features=ds.feature_names_numerical, method='mean',
                  add_indicators=True)

# Impute categorical features, and add an indicator column for missing values
ds.impute_missing(features=ds.feature_names_categorical, method='most_frequent',
                  add_indicators=True)

# Impute mixed features, and add an indicator column for missing and special values
# The list of special values need to be configured here manually.
ds.impute_missing(features=ds.feature_names_mixed, method='mean',
                  add_indicators=True, special_values=["SV1", "SV2"])

# %%
# Encoding categorical  features
ds.encode_categorical(features=("EDUCATION", "SEX", "MARRIAGE"), method="onehot")

# %%
# Scaling numerical features
ds.scale_numerical(features=("PAY_1", "PAY_2"), method="minmax")
ds.scale_numerical(features=("LIMIT_BAL", ), method="log1p")
ds.scale_numerical(features=("AGE", ), method="square")
ds.scale_numerical(features=("PAY_AMT1", ), method="quantile")
ds.scale_numerical(features=("PAY_1", "PAY_2",), method="log1p")

# %%
# Binning numerical features
ds.bin_numerical(features=("AGE", "PAY_3", ), bins=10)

# %%
# Execute the preprocessing steps defined above
ds.preprocess()
ds.to_df()

# %%
# Split data, set target, sample weight variables, and disable features that will not be used for modeling
ds.set_random_split()
ds.set_target("FlagDefault")
ds.set_sample_weight("LIMIT_BAL")
ds.set_inactive_features(features=('SEX_2.0',
                                   'MARRIAGE_1.0',
                                   'MARRIAGE_2.0'))
ds.feature_names, ds.feature_types, ds.train_x, ds.train_y, ds.test_x, ds.test_y

# %%
# Register data into MLFlow and view registered data
ds.register(override=True)
ds.list_registered_data()

# %%
# Load data from MLFlow
dsload = DataSet(name="TW-Credit")
dsload.load_registered_data(name="TW-Credit")
dsload
