"""
========================================
Other Plot
========================================

"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install mocharts

# %%
# Bar Stem Plot

import mocharts as mc
from IPython.display import HTML

x = ['a', 'b', 'c', 'd', 'e']
y = [0.1, 0.3, 0.2, -0.4, -0.5]
barstem = mc.barstemplot(x=x[::-1], y=y[::-1], stem_value=[-0.6, 0.2, 0, -0.1, 0.1], symmetric=True, value=[0.22, 0.33, 0.44, 0.55, 0.66])
barstem.set_xaxis(axis_name='x')
barstem.set_title('Bar Stem Plot')
# barstem.show()
HTML(barstem.show(return_html=True, silent=True))

# %%
# Tree Plot

tree_diagram = {'name': 'parant', 'children':[
                    {'name': 'children1'},
                    {'name': 'children2', 'children':[
                        {'name': 'children3'},
                        {'name': 'children4'}
                    ]}] }
tree = mc.treeplot(tree_diagram)
# tree.show()
HTML(tree.show(return_html=True, silent=True))

# %%
# Network Plot
import itertools
import networkx as nx

subset_sizes = [5, 4, 3, 2, 4, 3]

def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G

G = multilayered_graph(*subset_sizes)
pos = nx.multipartite_layout(G, subset_key="layer")

graph = mc.graphplot(G, layout = pos, node_size=20, edge_size=4)
# graph.show()
HTML(graph.show(return_html=True, silent=True))

# %%
# Radar Plot
x = ['ACC', 'AUC', 'F1', 'PRECISION', 'RECALL', 'ACC', 'AUC', 'F1', 'PRECISION', 'RECALL']
y = [0.8, 0.7, 0.8, 0.7, 0.6, 0.7, 0.6, 0.6, 0.5, 0.9]
label = ['Model1', 'Model1', 'Model1', 'Model1', 'Model1', 'Model2', 'Model2', 'Model2', 'Model2', 'Model2']

radar = mc.radarplot(x=x, y=y, label=label)
radar.set_tooltip()
radar.set_legend()
radar.set_grid(show=False)
# radar.show()
HTML(radar.show(return_html=True, silent=True))

# %%
# Parallel Plot
data = [[0.8, 0.7, 0.8, 0.7, 0.6], [ 0.7, 0.6, 0.6, 0.5, 0.9]]
label = ['Model1', 'Model2']
axis_list = [
            { 'dim': 0, 'name': 'ACC' },
            { 'dim': 1, 'name': 'AUC' },
            { 'dim': 2, 'name': 'F1' },
            { 'dim': 3, 'name': 'PRECISION' },
            { 'dim': 4, 'name': 'RECALL' }]
parallel = mc.parallel(data=data, axis_list=axis_list, labels=label)
parallel.set_grid(show=False)
parallel.set_tooltip(formatter = '{b}')
# parallel.show()
HTML(parallel.show(return_html=True, silent=True))
