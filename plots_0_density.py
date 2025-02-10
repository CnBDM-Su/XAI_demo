"""
========================================
Density Plot
========================================

"""

# %%
# Simple Density

from IPython.display import HTML
import mocharts as mc
import seaborn as sns
penguins = sns.load_dataset("penguins")
penguins=penguins.dropna()
kde = mc.kdeplot(x=penguins["flipper_length_mm"].values, color='red')
kde.set_tooltip(precision=4)
# kde.show()
HTML(kde.show(return_html=True, silent=True))

# %%
# Ridge Plot
import mocharts as mc
import seaborn as sns
penguins = sns.load_dataset("penguins")
penguins=penguins.dropna()
kde = mc.ridgeplot(x=penguins["flipper_length_mm"].values, label=penguins['species'].values)
kde.show()
HTML(kde.show(return_html=True, silent=True))

# %%
# Custom Density Plot
import mocharts as mc
import seaborn as sns
penguins = sns.load_dataset("penguins")
penguins=penguins.dropna()
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

import mocharts as mc
from sklearn.datasets import load_iris

data = load_iris()
target = [data['target_names'][i] for i in data['target']]

plot = mc.kde2Dplot(x=data['data'][:,0],y=data['data'][:,1], show_scatter=True,
             levels=8, bandwidth=1.2, threshold=0, color=['grey'])
plot.set_tooltip(show=False)

plot.set_title('kde2d')
plot.set_xaxis(axis_name=data['feature_names'][0])
plot.set_yaxis(axis_name=data['feature_names'][1])
# plot.show()
HTML(plot.show(return_html=True, silent=True))
