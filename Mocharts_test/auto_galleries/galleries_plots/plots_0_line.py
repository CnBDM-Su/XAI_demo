"""
========================================
Line Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Simple Line

import numpy as np
import mocharts as mc
from IPython.display import HTML

x = np.array(['Fri', 'Fri', 'Sat', 'Sat', 'Sun', 'Sun','Thur','Thur'])
y = np.array([16, 15, 20, 21, 12, 14, 20, 22])
label = np.array([0,1, 0,1,0,1,0,1])

line = mc.lineplot(x, y, label=label, color='red')
# line.show()
HTML(line.show(return_html=True, silent=True))

# %%
# Time Series

data = [["2000-06-05", 116], ["2000-06-06", 129], ["2000-06-07", 135], ["2000-06-08", 86], ["2000-06-09", 73], ["2000-06-10", 85], ["2000-06-11", 73], ["2000-06-12", 68], ["2000-06-13", 92], ["2000-06-14", 130], ["2000-06-15", 245], ["2000-06-16", 139], ["2000-06-17", 115], ["2000-06-18", 111], ["2000-06-19", 309], ["2000-06-20", 206], ["2000-06-21", 137], ["2000-06-22", 128], ["2000-06-23", 85], ["2000-06-24", 94], ["2000-06-25", 71], ["2000-06-26", 106], ["2000-06-27", 84], ["2000-06-28", 93], ["2000-06-29", 85], ["2000-06-30", 73], ["2000-07-01", 83], ["2000-07-02", 125], ["2000-07-03", 107], ["2000-07-04", 82], ["2000-07-05", 44], ["2000-07-06", 72], ["2000-07-07", 106], ["2000-07-08", 107], ["2000-07-09", 66], ["2000-07-10", 91], ["2000-07-11", 92], ["2000-07-12", 113], ["2000-07-13", 107], ["2000-07-14", 131], ["2000-07-15", 111], ["2000-07-16", 64], ["2000-07-17", 69], ["2000-07-18", 88], ["2000-07-19", 77], ["2000-07-20", 83], ["2000-07-21", 111], ["2000-07-22", 57], ["2000-07-23", 55], ["2000-07-24", 60]];
time = mc.lineplot(np.array(data)[:,0].ravel(), np.array(data)[:,-1].astype(int).ravel(), fill_area='blue')
time.set_grid(bottom='10%')
time.set_datazoom(show=True)
# time.show()
HTML(time.show(return_html=True, silent=True))

# %%
# Step Plot
x = np.array(['Thur', 'Fri', 'Sat',  'Sun'])
y = np.array([16, 15, 20, 21])

line = mc.lineplot(x, y, step='end')
# line.show()
HTML(line.show(return_html=True, silent=True))

# %%
# Custom Line Plot

x = np.array(['Fri', 'Fri', 'Sat', 'Sat', 'Sun', 'Sun','Thur','Thur'])
y = np.array([16, 15, 20, 21, 12, 14, 20, 22])
label = np.array([0,1, 0,1,0,1,0,1])

line = mc.lineplot(x, y, label=label, line_type='dashed', symbol='emptycircle', symbol_size=10)
line.set_legend()
line.set_xaxis(axis_name='week')
line.set_yaxis(axis_name='y')
line.set_title('Line Plot')
# line.show()
HTML(line.show(return_html=True, silent=True))