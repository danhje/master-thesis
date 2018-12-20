import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D  
import nest.topology as topo
source_layer = topo.CreateLayer({'elements': 'iaf_neuron',                                 'positions': [(0.0, 0.0)]})
x = np.random.uniform(-0.5, 0.5, 10000)y = np.random.uniform(-0.5, 0.5, 10000)pos = zip(x, y)
target_layer = topo.CreateLayer({'elements': 'iaf_neuron',                                 'positions': pos})
mask = {'rectangular': {'lower_left': [-0.5, -0.5],                        'upper_right': [0.5, 0.5]}}kernel = {'gaussian': {'p_center': 1., 'sigma': .2}}conn_specs = {'connection_type': 'convergent',              'mask': mask, 'kernel': kernel}topo.ConnectLayers(source_layer, target_layer, conn_specs)import matplotlib.pyplot as plt
center = topo.FindCenterElement(source_layer)
fig = topo.PlotLayer(target_layer, nodecolor='grey', nodesize=3)
topo.PlotTargets(center, target_layer, fig=fig, tgt_color='red', tgt_size=5)

ax = plt.gca()

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()

filename_base = __file__.split('.')[0]
plt.savefig(filename_base + '.png', bbox_inces='tight', pad_inches=0)

