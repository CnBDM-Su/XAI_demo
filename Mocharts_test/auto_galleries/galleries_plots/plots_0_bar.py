"""
========================================
Bar Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple Bar

import numpy as np
import mocharts as mc
from IPython.display import HTML

x = np.array(['Thur', 'Thur', 'Fri', 'Fri', 'Sat', 'Sat', 'Sun', 'Sun'])
y = np.array([16, 15, 20, 21, 12, 14, 20, 22])
label = np.array([0,1,0,1,0,1,0,1])

bar = mc.barplot(x, y, label=label, color=['red','blue'])
bar.set_legend()
bar.set_figsize((8,4))
# bar.show()
HTML(bar.show(return_html=True, silent=True))

# %%
# Bar Plot with custom bar color

x = np.array(['Thur',  'Fri', 'Sat',  'Sun', ])
y = np.array([16, 15, 20, 21])

bar = mc.barplot(x, y, bar_color=['red','blue','green','yellow',])
bar.set_figsize((8,4))
# bar.show()
HTML(bar.show(return_html=True, silent=True))

# %%
# Horizontal Bar Plot

x = np.array(['long string 1', 'long string 1', 'long string 2', 'long string 2',
              'long string 3', 'long string 3', 'long string 4', 'long string 4'])
y = np.array([16, 15, 20, 21, 12, 14, 20, 22])
label = np.array([0,1,0,1,0,1,0,1])

bar = mc.barplot(x, y, label=label, orient='horizontal')
bar.set_legend()
bar.set_figsize((8,4))
# bar.show()
HTML(bar.show(return_html=True, silent=True))

# %%
# Custom Bar Plot

x = np.array(['Thur', 'Thur', 'Fri', 'Fri', 'Sat', 'Sat', 'Sun', 'Sun'])
y = np.array([1625, 1535, 2202, 211, 1222, 1114, 2205, 2522])
label = np.array([0,1,0,1,0,1,0,1])

bar = mc.barplot(x, y, label=label, show_label='inside', multiple='stack', bar_width='60%')
# bar.set_legend()
bar.set_xaxis(axis_name='week')
bar.set_yaxis(axis_name='count')
bar.set_title('Bar Plot')
bar.set_figsize((6,4))
# bar.show()
HTML(bar.show(return_html=True, silent=True))