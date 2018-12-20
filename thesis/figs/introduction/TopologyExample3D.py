import matplotlib.pyplot as plt
import nest
import nest.topology as topo



import numpy as np
x = np.random.uniform(-0.5, 0.5, 1000)
y = np.random.uniform(-0.5, 0.5, 1000)
pos = zip(x, y)
ls = topo.CreateLayer({'elements': 'iaf_neuron', 'positions': [(0.0, 0.0)]})
lt = topo.CreateLayer({'elements': 'iaf_neuron', 'positions': pos})
mask = {'rectangular': {'lower_left': [-0.4, -0.4], 
                        'upper_right': [0.4, 0.4]}}
kernel = {'gaussian': {'p_center': 1., 'sigma': 0.2,
                         'mean': 0., 'c': 0.}}
conndict = {'connection_type': 'divergent',
            'mask': mask, 'kernel': kernel}
topo.ConnectLayers(ls, lt, conndict)

fig = topo.PlotLayer(lt, nodecolor='grey')

driver = topo.FindCenterElement(ls)
topo.PlotTargets(driver, lt, fig=fig,
                 mask=mask, kernel=kernel,
                 mask_color='purple',
                 kernel_color='blue',
                 src_size=50, src_color='red',
                 tgt_size=20, tgt_color='red')

a = fig.gca()

#a.w_zaxis.tick_top()
#a.set_frame_on(False)
#a.set_axis_off()

#a.xaxis.set_visible(False)
#a.yaxis.set_visible(False)

#a.zaxis.set_label_position('bottom')

#a.set_xticks([])
#a.set_yticks([])
#a.set_zticks([])

#a.set_xticklabels(())
#a.set_yticklabels(())
#a.set_zticklabels(())

#a.xaxis.set_ticklabels([])
#a.yaxis.set_ticklabels([])

#a.xaxis.set_ticks_position('none')
#a.yaxis.set_ticks_position('none')

#a.xaxis.set_tick_params(length=-10, width=-10, zorder=-10**1000, colors='0.89')
#a.yaxis.set_tick_params(length=-10, width=-10, zorder=-10, colors='0.89')


#plt.draw()

plt.tight_layout()
plt.savefig('TopologyExample3D.pdf', bbox_inches='tight', pad_inches=0)

