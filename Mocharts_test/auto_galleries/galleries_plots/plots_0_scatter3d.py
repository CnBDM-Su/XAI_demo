"""
========================================
3D Scatter Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple 3D Scatter

import mocharts as mc
from sklearn.datasets import load_iris
from IPython.display import HTML

iris = load_iris()
X = iris.data
y = iris.target

fig = mc.scatter3Dplot(x=X[:,0], y=X[:,1], z=X[:,2], color='orange')
fig.set_figsize((6,6))
# fig.show()
HTML(fig.show(return_html=True, silent=True))


# %%
# Custom 3D Scatter Plot

fig = mc.scatter3Dplot(x=X[:,0], y=X[:,1], z=X[:,2], label=y, symbol_size=15, show_label=True,
                       label_distance_ratio=2., symbol='arrow', label_order=[2,1,0], color=['yellow','purple','pink'])

fig.set_title('3D Scatter')
fig.set_legend()
fig.set_xaxis(axis_name=iris.feature_names[0])
fig.set_yaxis(axis_name=iris.feature_names[1])
fig.set_zaxis(axis_name=iris.feature_names[2])
fig.set_grid(alpha=20, beta=20, distance=200)
fig.set_figsize((5,5))
# fig.show()
HTML(fig.show(return_html=True, silent=True))
