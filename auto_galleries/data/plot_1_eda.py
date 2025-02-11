"""
========================================
Exploratory Data Analysis
========================================

"""

# %%
# Import modeva modules
from modeva import DataSet

# %%
# Load TaiwanCredit Dataset
ds = DataSet()
ds.load("TaiwanCredit")

# %%
# Data summary
result = ds.summary()
result.table["summary"]

# %%
# Data summary results for numerical variables
result.table["numerical"]

# %%
# Data summary results for categorical variables
result.table["categorical"]

# %%
# EDA 1D by density
result = ds.eda_1d(feature="PAY_1")
result.plot()

# %%
# EDA 1D by histogram
result = ds.eda_1d(feature="BILL_AMT1", plot_type="histogram")
result.plot()

# %%
# EDA 2D
result = ds.eda_2d(feature_x="BILL_AMT1", feature_y="PAY_1", sample_size=1000)
result.plot()

# %%
# EDA 2D with color and smoothing curve
result = ds.eda_2d(feature_x="BILL_AMT1", feature_y="BILL_AMT2", feature_color="SEX", sample_size=1000, 
                   smoother_order=2)
result.plot(figsize=(6, 5))

# %%
# EDA 2D between numerical and categorical variables
result = ds.eda_2d(feature_x="SEX", feature_y="BILL_AMT1")
result.plot()

# %%
# EDA 2D between two categorical and categorical variables
result = ds.eda_2d(feature_x="MARRIAGE", feature_y="SEX")
result.plot()

# %%
# EDA 3D
result = ds.eda_3d(feature_x="SEX", feature_y="PAY_1", feature_z="BILL_AMT1", feature_color="EDUCATION",
                   sample_size=1000)
result.plot()

# %%
# Correlation
result = ds.eda_correlation(features=('PAY_1',
                                      'PAY_2',
                                      'PAY_3',
                                      'PAY_4',
                                      'PAY_5',
                                      'PAY_6'),
                            dataset="main", sample_size=10000)
result.plot()

# %%
# XICorrelation
# 1. XiCor detects both linear and nonlinear dependencies between continuous variables.
# 2. It typically ranges from 0 (no dependence) to 1 (strong dependence), providing a more comprehensive view of relationships.
# 3. Negative XI correlation does not have any innate significance, other than close to zero.

result = ds.eda_correlation(features=('PAY_1',
                                      'PAY_2',
                                      'PAY_3',
                                      'PAY_4',
                                      'PAY_5',
                                      'PAY_6'),
                            dataset="main", method="xicor", sample_size=10000) # "pearson", "spearman", "kendall", "xicor"
result.plot()

# %%
# PCA
result = ds.eda_pca(features=("EDUCATION",
                              "MARRIAGE",
                              'PAY_1',
                              'PAY_2',
                              'PAY_3',
                              'PAY_4',
                              'PAY_5',
                              'PAY_6'),
                    n_components=10, dataset="main", sample_size=None)
result.plot()

# # %%
# # Umap
result = ds.eda_umap(features=('PAY_1',
 'PAY_2',
 'PAY_3',
 'PAY_4',
 'PAY_5',
 'PAY_6'), n_components=2, dataset="main", sample_size=1000)
result.table
