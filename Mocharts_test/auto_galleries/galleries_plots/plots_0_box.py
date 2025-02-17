"""
========================================
Box Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple Box

from IPython.display import HTML
import mocharts as mc
import pandas as pd
tips = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
box = mc.boxplot(x=tips["total_bill"].values, orient='vertical', color='orange')
box.set_tooltip()
# box.show()
HTML(box.show(return_html=True, silent=True))

# %%
# Horizontal Box Plot

box = mc.boxplot(x=tips["day"].values, y=tips["total_bill"].values, orient='horizontal', color='orange')
box.set_tooltip
# box.show()
HTML(box.show(return_html=True, silent=True))

# %%
# Custom Box Plot

box = mc.boxplot(x=tips["day"].values, y=tips["total_bill"].values, label=tips["smoker"].values, orient='vertical', outliers=True, color=['orange','green'])
box.set_tooltip()
box.set_xaxis(axis_name='day')
box.set_yaxis(axis_name='y')
box.set_title('Box Plot')
box.set_legend(title='label')
# box.show()
HTML(box.show(return_html=True, silent=True))
