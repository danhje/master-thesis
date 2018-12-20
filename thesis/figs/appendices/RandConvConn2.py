import numpy as np
import numpy.random as rnd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D   

fig = plt.figure(figsize=(12, 5), frameon=False)
ax = fig.gca()
ax.set_axis_off()

def node_pos(n):
    if 0 < n <= 10: return (n*10-9, 10, -3)
    else: return (n*10-100-9, 40, 3)

def draw_node(x, y, font_offset, n):
    c = Circle((x, y), radius=.5, facecolor='black', linestyle='solid', edgecolor='black', 
               linewidth=1, zorder=1)
    ax.add_patch(c)
    ax.text(x, y+font_offset, '$' + str(n) + '$', fontsize=20, horizontalalignment='center', 
            verticalalignment='center')

def draw_arrow(start, stopp):
    dx = float(stopp[0])-start[0]; dy = float(stopp[1])-start[1]
    if dx > 0:
        alpha = np.arctan(dy/dx)
    elif dx < 0:
        alpha = np.pi - np.arctan(dy/-dx)
    else:
        alpha = np.pi / 2.0
    h = 2.0
    print alpha, dx, dy
    cx = h * np.cos(alpha)
    cy = h * np.sin(alpha)
    x = start[0]+cx
    y = start[1]+cy
    dx = dx - 2*cx
    dy = dy - 2*cy
    ax.arrow(x, y, dx, dy, shape='full', head_width=1.2, head_length=2.0, 
             fc="k", ec="k", length_includes_head="True")


rnd.seed(12345)
random.seed(12345)

sources = range(1, 11)
targets = range(11, 21)
nodes = sources + targets
for node in nodes:
    x, y, off = node_pos(node)
    draw_node(x, y, off, node)

for target in targets:
    drawn = []
    for i in range(3):
        source = random.choice(sources)
        sx = node_pos(source)[0]; sy = node_pos(source)[1]
        tx = node_pos(target)[0]; ty = node_pos(target)[1]
        if source in drawn:
            draw_arrow((sx+0.4, sy), (tx+0.4, ty))

        else:
            draw_arrow((sx, sy), (tx, ty))
        drawn.append(source)

ax.text(103, 10, '$\mathrm{Source}$\n$\mathrm{nodes}$', fontsize=22, horizontalalignment='center', 
        verticalalignment='center')
ax.text(103, 40, '$\mathrm{Target}$\n$\mathrm{nodes}$', fontsize=22, horizontalalignment='center', 
        verticalalignment='center')

plt.xlim(0, 110)
plt.ylim(0, 50)
#plt.show(block=True)
plt.tight_layout()
plt.savefig('RandConvConn2.pdf', bbox_inches='tight', pad_inches=0)
plt.cla()
plt.close('all')
