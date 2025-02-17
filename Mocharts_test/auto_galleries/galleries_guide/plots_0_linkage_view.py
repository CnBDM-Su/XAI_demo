"""
========================================
Linkage View
========================================

Mocharts supports linked views for multiple charts. In this example, we demonstrate how manipulating one chart can control the display of another related chart.
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
data = load_iris()
target = np.array([data['target_names'][int(i)] for i in data['target']]).reshape(-1,1)
df = pd.DataFrame(np.concatenate([data['data'], target.astype(object)], axis=1), columns=data['feature_names']+['Species'])
for feature in data['feature_names']:
    df[feature] = df[feature].astype(float)

# %%
# View the mean value of samples in selected category

import mocharts as mc
from IPython.display import HTML

def func1(js_input, py_input):  # for use by card2
    idx = js_input['selected']
    fig2 = mc.histplot(x=df.iloc[:, 0][df.Species == idx].values, bins=8)
    fig2.chart_id = py_input['link_id']
    fig2.set_title(f"Histogram of species {idx}")
    fig2.set_tooltip(precision=4)
    fig2.set_figsize(figsize=(5, 5))
    fig2.set_yaxis(axis_name='Count')
    fig2.set_xaxis(axis_name=df.columns[0])
    return fig2.show(return_html=True, silent=True)

tmp = df.Species.value_counts()
fig = mc.barplot(x=list(tmp.index.astype(str)), y=tmp.values, sorted_x=True)
fig.set_event(type_='click', func=func1, link_id='card2')    # Link to ChartID = card2, 
fig.set_yaxis(axis_name='Count', nameGap=20)
fig.set_xaxis(axis_name='Species')
fig.set_figsize((4, 4))
fig.set_title('Histogram of species')
html1 = fig.show(return_html=True, silent=True)

fig2 = mc.histplot(x=df.iloc[:,0].values, bins=8)
fig2.chart_id = 'card2'
fig2.set_title("Histogram")
fig2.set_tooltip(precision=4)
fig2.set_figsize(figsize=(4, 4))
fig2.set_yaxis(axis_name='Count')
fig2.set_xaxis(axis_name=df.columns[0])
html2 = fig2.show(return_html=True, silent=True)

html_content = f"""
<div style="display: flex; justify-content: space-around;">
    <div><p>{html1}</p></div>
    <div><p>{html2}</p></div>
</div>
"""

HTML(html_content)

# %%
# View density plot of seleted samples
def func1(js_input, py_input):  # for use by card2
    selected = js_input['selected']
    if len(selected) != 0:
        link_id = py_input['link_id']
        data_selected = df.iloc[selected, list(df.columns).index('Species')]
        tmp = data_selected.value_counts()
        fig = mc.barplot(list(tmp.index), tmp.values)
        fig.link_id = link_id
        fig.set_xaxis(axis_name='Species')
        fig.set_tooltip(precision=4)
        fig.set_title('Species of selected samples')
        fig.set_figsize(figsize=(4,4))
        return fig.show(silent=True, return_html=True)

sca1 = mc.scatterplot(x=df['sepal length (cm)'], y=df['sepal width (cm)'], allow_large=False)
sca1.set_xaxis(axis_name='sepal length (cm)')
sca1.set_yaxis(axis_name='sepal width (cm)')
sca1.set_event(type_='brushselected', func=func1, link_id='related_counts')
sca1.set_brush(toolbox=['rect', 'polygon', 'keep', 'clear'])
sca1.set_title('Scatter for samples selection')
sca1.set_figsize((4, 4))
html3 = sca1.show(return_html=True, silent=True)

tmp = df['Species'].value_counts()
fig = mc.barplot(list(tmp.index), tmp.values)
fig.chart_id='related_counts'
fig.set_figsize(figsize=(4,4))
fig.set_xaxis(axis_name='Species')
fig.set_tooltip(precision=4)
fig.set_title('Species of selected samples')
html4 = fig.show(return_html=True, silent=True)

html_content2 = f"""
<div style="display: flex; justify-content: space-around;">
    <div><p>{html3}</p></div>
    <div><p>{html4}</p></div>
</div>
"""

HTML(html_content2)