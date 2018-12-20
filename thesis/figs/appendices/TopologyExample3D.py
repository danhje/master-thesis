import matplotlib.pyplot as plt
import nest
import nest.topology as topo



import numpy as np
x = np.random.uniform(-0.5, 0.5, 1000)
y = np.random.uniform(-0.5, 0.5, 1000)
z = np.random.uniform(-0.5, 0.5, 1000)
pos = zip(x, y, z)
l = topo.CreateLayer({'elements': 'iaf_neuron',
                      'positions': pos})
fig = topo.PlotLayer(l)

a = fig.gca()

#a.w_zaxis.tick_top()
a.set_frame_on(False)
#a.set_axis_off()

a.w_xaxis.set_visible(False)
a.w_yaxis.set_visible(False)
a.w_zaxis.set_visible(False)

#a.zaxis.set_label_position('bottom')

#a.set_xticks([])
#a.set_yticks([])
#a.set_zticks([])

#a.set_xticklabels(())
#a.set_yticklabels(())
#a.set_zticklabels(())

a.w_xaxis.set_ticklabels([])
a.w_yaxis.set_ticklabels([])
a.w_zaxis.set_ticklabels([])

a.w_xaxis.set_ticks_position('none')
a.w_yaxis.set_ticks_position('none')
a.w_zaxis.set_ticks_position('none')

a.w_xaxis.set_tick_params(length=-10, width=-10, zorder=-10**1000, colors='0.89')
a.w_yaxis.set_tick_params(length=-10, width=-10, zorder=-10, colors='0.89')
a.w_zaxis.set_tick_params(length=-10, width=-10, zorder=-10, colors='0.89')


#plt.draw()

plt.tight_layout()
plt.savefig('TopologyExample3D.pdf', bbox_inches='tight', pad_inches=0)

