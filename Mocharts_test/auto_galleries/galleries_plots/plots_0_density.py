"""
========================================
Density Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple Density

from IPython.display import HTML
import mocharts as mc
import pandas as pd
penguins = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')
penguins=penguins.dropna()
kde = mc.kdeplot(x=penguins["flipper_length_mm"].values, color='red')
kde.set_tooltip(precision=4)
# kde.show()
HTML(kde.show(return_html=True, silent=True))

# %%
# Ridge Plot

kde = mc.ridgeplot(x=penguins["flipper_length_mm"].values, label=penguins['species'].values)
# kde.show()
HTML(kde.show(return_html=True, silent=True))

# %%
# Custom Density Plot

kde = mc.kdeplot(x=penguins["flipper_length_mm"].values, label=penguins['species'].values, common_norm=True, fill_area=False,
                color=['red','yellow','pink'])
kde.set_tooltip(precision=4)
kde.set_xaxis(axis_name='flipper_length_mm')
kde.set_yaxis(axis_name='density')
kde.set_title('Density')
kde.set_legend()
# kde.show()
HTML(kde.show(return_html=True, silent=True))

# %%
# 2D KDE Plot

plot = mc.kde2Dplot(x=penguins["flipper_length_mm"].values,
                    y=penguins["bill_depth_mm"].values, show_scatter=True,
                    levels=8, bandwidth=1.2, threshold=0, color=['grey'])
plot.set_tooltip(show=False)
plot.set_title('kde2d')
plot.set_xaxis(axis_name='flipper_length_mm')
plot.set_yaxis(axis_name='bill_depth_mm')
# plot.show()
HTML(plot.show(return_html=True, silent=True))
