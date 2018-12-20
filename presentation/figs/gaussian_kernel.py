import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D  
import nest.topology as topo

x = np.random.uniform(-0.5, 0.5, 10000)
target_layer = topo.CreateLayer({'elements': 'iaf_neuron',

center = topo.FindCenterElement(source_layer)
fig = topo.PlotLayer(target_layer, nodecolor='grey', nodesize=3)
topo.PlotTargets(center, target_layer, fig=fig, tgt_color='red', tgt_size=5)

ax = plt.gca()

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()

filename_base = __file__.split('.')[0]
plt.savefig(filename_base + '.png', bbox_inces='tight', pad_inches=0)
