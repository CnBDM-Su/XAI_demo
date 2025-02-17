"""
========================================
Basic Plots
========================================

Mocharts supports a variety of chart types. In this example, we will demonstrate some simple use cases for these charts. For more details, you can refer to the API documentation for each plot or explore the examples for each chart.
"""
# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts


# %%
# Get Iris dataset

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from IPython.display import HTML
data = load_iris()
target = np.array([data['target_names'][int(i)] for i in data['target']]).reshape(-1,1)
df = pd.DataFrame(np.concatenate([data['data'], target.astype(object)], axis=1), columns=data['feature_names']+['Species'])
for feature in data['feature_names']:
    df[feature] = df[feature].astype(float)

# %%
# Scatter Plot

import mocharts as mc
plot = mc.scatterplot(x=df['sepal length (cm)'].values, y=df['sepal width (cm)'].values, label=df['Species'].values,
                      allow_large=False, sampling=True)
plot.set_xaxis(axis_name='sepal length (cm)')
plot.set_yaxis(axis_name='sepal width (cm)')
plot.set_legend(title='Species', top='13%', left='86%')
plot.set_title('Scatter Plot')
plot.set_grid(right='15%')
plot.set_figsize((6,6))
# plot.show()
HTML(plot.show(return_html=True, silent=True))

# %%
# Line Plot
category = pd.qcut(df['sepal width (cm)'].astype(float), 8).astype(object)
category.name = 'Sepal Width Range'
new_data = pd.concat([df, category], axis=1)
mean_val = new_data.groupby('Sepal Width Range').mean(numeric_only=True).loc[:,'sepal length (cm)']

line = mc.lineplot(x=list(mean_val.index.astype(str)), y=mean_val.values)
line.set_tooltip(precision=4)
line.set_xaxis(axis_name='Sepal Width Range', axisLabel_rotate=30, nameGap=50)
line.set_yaxis(axis_name='Mean sepal length (cm)')
line.set_title('Line Plot')
line.set_figsize((6,6))
# line.show()
HTML(line.show(return_html=True, silent=True))

# %%
# Bar Plot
mean_length = df.groupby('Species').mean().loc[:, 'sepal length (cm)']

bar = mc.barplot(x=list(mean_length.index), y=np.round(mean_length.values,4), show_label='top')
bar.set_xaxis(axis_name='Species')
bar.set_yaxis(axis_name="Mean sepal length (cm)")
bar.set_title('Bar Plot')
bar.set_tooltip(precision=4)
bar.set_figsize((6,6))
# bar.show()
HTML(bar.show(return_html=True, silent=True))

# %%
# Box Plot
box = mc.boxplot(x=df['Species'], y=df['sepal length (cm)'], orient='vertical')
box.set_xaxis(axis_name='Species')
box.set_yaxis(axis_name="sepal length (cm)")
box.set_title('Box Plot')
box.set_tooltip(precision=4)
box.set_figsize((6,6))
# box.show()
HTML(box.show(return_html=True, silent=True))


# %%
# Histogram Plot
hist = mc.histplot(x=df['sepal length (cm)'], label=df['Species'], bins=15)
hist.set_xaxis(axis_name='sepal length (cm)')
hist.set_yaxis(axis_name='Count')
hist.set_legend()
hist.set_title('Histogram Plot')
hist.set_figsize((6,6))
# hist.show()
HTML(hist.show(return_html=True, silent=True))

# %%
# Density Plot
kde = mc.kdeplot(x=df['sepal length (cm)'], label=df['Species'])
kde.set_xaxis(axis_name='sepal length (cm)')
kde.set_yaxis(axis_name='Density')
kde.set_title('Density Plot')
kde.set_legend()
kde.set_tooltip(precision=4)
kde.set_figsize((6,6))
# kde.show()
HTML(kde.show(return_html=True, silent=True))

# %%
# Heatmap Plot
convert = {'setosa':0, 'versicolor':1, 'virginica':2}
new_data = df.Species.apply(lambda x: convert[x])
new_data = pd.concat([df.drop('Species', axis=1), new_data], axis=1)
corr = new_data.corr().round(4)
heatmap = mc.heatmap(corr.values, row_names=list(corr.columns), col_names=list(corr.columns), show_label=True)
heatmap.set_xaxis(axisLabel_rotate=30)
heatmap.set_tooltip()
heatmap.set_title('Correlation Heatmap', left='55%')
heatmap.set_figsize((6,6))
heatmap.set_visualmap(calculable=True, width=15, height=520)
# heatmap.show()
HTML(heatmap.show(return_html=True, silent=True))