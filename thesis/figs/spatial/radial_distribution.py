import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D  
import nest.topology as topo
source_layer = topo.CreateLayer({'elements': 'iaf_neuron',
                                 'positions': [(0.0, 0.0)]})
x = np.random.uniform(-0.5, 0.5, 1000)
y = np.random.uniform(-0.5, 0.5, 1000)
pos = zip(x, y)
target_layer = topo.CreateLayer({'elements': 'iaf_neuron',
                                 'positions': pos})
mask = {'rectangular': {'lower_left': [-0.5, -0.5],
                        'upper_right': [0.5, 0.5]}}
kernel = {'gaussian': {'p_center': 1., 'sigma': .2}}
conn_specs = {'connection_type': 'convergent',
              'mask': mask, 'kernel': kernel}
              
topo.ConnectLayers(source_layer, target_layer, conn_specs)

center = topo.FindCenterElement(source_layer)
fig = topo.PlotLayer(target_layer, nodecolor='black', nodesize=3)
#topo.PlotTargets(center, target_layer, fig=fig, tgt_color='red', tgt_size=5)

ax = plt.gca()
c = Circle((0, 0), radius=0.27, facecolor='none', linestyle='solid', edgecolor='red', linewidth=1, zorder=2)
ax.add_patch(c)
c = Circle((0, 0), radius=0.4, facecolor='none', linestyle='solid', edgecolor='red', linewidth=1, zorder=2)
ax.add_patch(c)

ax.arrow(0.0, 0.0, 0.186, 0.186, shape='full', head_width=0.03, head_length=0.03, fc="red", ec="red", length_includes_head="True", linewidth=1)
ax.text(0.09, 0.07, '$D$', fontsize=24, color='red', horizontalalignment='left', verticalalignment='center')

ax.arrow(0.22+2*0.067, 0.22+2*0.067, -0.067, -0.067, shape='full', head_width=0.03, head_length=0.03, fc="red", ec="red", length_includes_head="True", linewidth=1)

ax.text(0.23, 0.21, '$dD$', fontsize=24, color='red', horizontalalignment='left', verticalalignment='center')


 

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.tight_layout()

filename_base = __file__.split('.')[0]
plt.savefig(filename_base + '.pdf', bbox_inces='tight', pad_inches=0)

