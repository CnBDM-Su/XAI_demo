"""
========================================
Histogram Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple Histogram

from IPython.display import HTML
import mocharts as mc
import pandas as pd
penguins = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')
penguins=penguins.dropna()
hist = mc.histplot(x=penguins["flipper_length_mm"].values, bins=20, color='orange')
# hist.show()
HTML(hist.show(return_html=True, silent=True))

# %%
# Custom Histogram Plot

hist = mc.histplot(x=penguins["flipper_length_mm"].values, label=penguins['species'].values, bins=20, color=['orange','green','grey'])
hist.set_title('Histogram Plot')
hist.set_xaxis(axis_name='flipper_length_mm')
hist.set_yaxis(axis_name='count')
hist.set_legend()
# hist.show()
HTML(hist.show(return_html=True, silent=True))
