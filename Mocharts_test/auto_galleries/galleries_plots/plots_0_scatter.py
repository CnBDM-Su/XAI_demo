"""
========================================
Scatter Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple Scatter

from IPython.display import HTML
import mocharts as mc
from sklearn.datasets import make_circles
pos,label = make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.8)
x,y = pos[:,0], pos[:,1]

plot = mc.scatterplot(x=x, y=y, color='orange')
# plot.show()
HTML(plot.show(return_html=True, silent=True))

# %%
# Custom Scatter Plot
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=400, noise=0.1, random_state=42)

plot = mc.scatterplot(x=X[:,0], y=X[:,1], label=y, symbol_size=15, show_label='serie', smoother_order=3, color=['orange','green'])
plot.set_legend()
plot.set_title('Moons')
plot.set_xaxis(axis_name='X')
plot.set_yaxis(axis_name='Y')
# plot.show()
HTML(plot.show(return_html=True, silent=True))
