import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D  
import nest.topology as topo

x = np.random.uniform(-0.5, 0.5, 10000)
target_layer = topo.CreateLayer({'elements': 'iaf_neuron',

center = topo.FindCenterElement(source_layer)
fig = topo.PlotLayer(target_layer, nodecolor='grey', nodesize=3)
#topo.PlotTargets(center, target_layer, fig=fig, tgt_color='red', tgt_size=5)

ax = plt.gca()
c = Circle((0, 0), radius=0.35, facecolor='none', linestyle='solid', edgecolor='red', linewidth=2, zorder=2)
ax.add_patch(c)
c = Circle((0, 0), radius=0.4, facecolor='none', linestyle='solid', edgecolor='red', linewidth=2, zorder=2)
ax.add_patch(c)

ax.arrow(0.2, 1.3, -0.15, -0.13, shape='full', head_width=0.05, head_length=0.07, fc="k", ec="k", length_includes_head="True")


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.tight_layout()

filename_base = __file__.split('.')[0]
plt.savefig(filename_base + '.png', bbox_inces='tight', pad_inches=0)
