import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D                        


fig = plt.figure(figsize=(6,6), frameon=False)


r = Rectangle((-1, -1), width=2, height=2, fill=False, edgecolor='black', linewidth=1, zorder=10)
c1 = Circle((0, 0), radius=1, facecolor='none', edgecolor='purple', linewidth=1, zorder=2)
c2 = Circle((0, 0), radius=1.2, facecolor='none', edgecolor='red', linewidth=1, zorder=2)
c3 = Circle((0, 0), radius=0.8, facecolor='none', edgecolor='blue', linewidth=1, zorder=2)
l1 = Line2D([-0.8,0], [0,0], color='blue', linewidth=1, zorder=3)
l2 = Line2D([-np.cos(np.pi / 8.), 0], [np.sin(np.pi / 8.),0], color='purple', linewidth=1, zorder=3)
l3 = Line2D([-1.2*np.cos(np.pi / 4.), 0], [1.2*np.sin(np.pi / 4.),0], color='red', linewidth=1, zorder=3)

x = 1; y = np.tan(np.arccos(1/1.2))

l4 = Line2D([0, x], [0, y], color='black', linestyle='--', linewidth=1, zorder=3)
l5 = Line2D([0, y], [0, x], color='black', linestyle='--', linewidth=1, zorder=3)

theta1 = np.arccos(1/1.2)*360/(2*np.pi)
theta2 = 90 - theta1
a1 = Arc(xy=(0, 0), width=2.35, height=2.35, theta1=theta1, theta2=theta2, angle=0, facecolor='none', edgecolor='black', linestyle='dashed', linewidth=1, zorder=3)


ax = plt.gca()
ax.set_axis_off()
ax.add_patch(r)
ax.add_patch(c1)
ax.add_patch(c2)
ax.add_patch(c3)
ax.add_line(l1)
ax.add_line(l2)
ax.add_line(l3)
#ax.add_line(l4)
#ax.add_line(l5)
#ax.add_patch(a1)

ax.text(-.7, .0, '$D_1$', fontsize=24, color='blue')
ax.text(-.62, .25, '$D_2$', fontsize=24, color='purple')
ax.text(-.47, .45, '$D_3$', fontsize=24, color='red')
ax.text(0, -1.14, '$L$', fontsize=24, color='black', horizontalalignment='center')
ax.text(1.02, 0, '$L$', fontsize=24, color='black', verticalalignment='center')


plt.ylim(-1.35, 1.35)
plt.xlim(-1.35, 1.35)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()


filename_base = __file__.split('.')[0]
plt.savefig(filename_base + '.pdf', bbox_inces='tight', pad_inches=0)
plt.show(block=True)

