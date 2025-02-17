"""
========================================
Container
========================================

Mocharts supports various layout options for displaying multiple plots. In this example, we will demonstrate how to display a single plot, how to combine multiple plots into a single figure, and how to manage multiple plots using the SubPlots object.
"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Figure (Single Plot)
from IPython.display import HTML
import mocharts as mc
from sklearn.datasets import make_circles

pos,label = make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.8)
x,y = pos[:,0], pos[:,1]

fig = mc.Figure()
plot = mc.scatterplot(x=x, y=y, label=label)
fig.add(plot)
# fig.show()
HTML(fig.show(return_html=True, silent=True))

# %%
# Figure (Multiple Plots)
fig = mc.Figure()

scatter = mc.scatterplot(x=x, y=y)
fig.add(scatter)

line = mc.lineplot([-1,1], [-1,1])
fig.add(line)
fig.set_title('Combine Plot')

# fig.show()
HTML(fig.show(return_html=True, silent=True))

# %%
# SubPlots

scatter = mc.scatterplot(x=x, y=y)
line = mc.lineplot([-1,1], [-1,1])
line.set_title('Line')
scatter.set_title('Scatter')
sub = mc.SubPlots([scatter, line], subplots_grid=[10,1], subplots_position=[[[0,5],[0,1]],[[6,10],[0,1]]])

# sub.show()
HTML(sub.show(return_html=True, silent=True))
