"""
========================================
Heatmap Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple Heatmap

from IPython.display import HTML
import mocharts as mc
import numpy as np

uniform_data = np.random.rand(10, 12)
hm = mc.heatmap(data=uniform_data)
hm.set_visualmap(height=560, width=20)
hm.set_figsize((3,3))
hm.set_toolbox(itemSize=30, left='7%')
# hm.show()
HTML(hm.show(return_html=True, silent=True))


# %%
# Custom Heatmap Plot
uniform_data = np.round(np.random.rand(10, 12), 3)
hm = mc.heatmap(data=uniform_data, extent=[0, 1, 0, 1], show_label=True)
hm.set_visualmap(calculable=True)
hm.set_xaxis(axis_name='x')
hm.set_yaxis(axis_name='y')
hm.set_title('Heatmap')
hm.set_figsize((4,4))
# hm.show()
HTML(hm.show(return_html=True, silent=True))
