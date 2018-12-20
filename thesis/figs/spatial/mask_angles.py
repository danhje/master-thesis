import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D                        


fig = plt.figure(figsize=(6,6), frameon=False)


r = Rectangle((-1, -1), width=2, height=2, fill=False, edgecolor='black', linewidth=1, zorder=5)
c = Circle((0, 0), radius=1.2, facecolor='none', linestyle='dashed', edgecolor='black', linewidth=1, zorder=2)
lh = Line2D([-1.35, 1.35], [0, 0], color='black', linestyle='-', linewidth=1, zorder=0)
lv = Line2D([0, 0], [-1.35, 1.35], color='black', linestyle='-', linewidth=1, zorder=0)


x = 1; y = np.tan(np.arccos(1/1.2))
l4 = Line2D([0, x], [0, y], color='black', linestyle='--', linewidth=1, zorder=3)
l5 = Line2D([0, y], [0, x], color='black', linestyle='--', linewidth=1, zorder=3)

t1 = np.arccos(1/1.2)*360/(2*np.pi)
t2 = 90 - t1
a1 = Arc(xy=(0, 0), width=2.4, height=2.4, theta1=0*90+t1, theta2=0*90+t2, linewidth=2, zorder=3)
a2 = Arc(xy=(0, 0), width=2.4, height=2.4, theta1=1*90+t1, theta2=1*90+t2, linewidth=2, zorder=3)
a3 = Arc(xy=(0, 0), width=2.4, height=2.4, theta1=2*90+t1, theta2=2*90+t2, linewidth=2, zorder=3)
a4 = Arc(xy=(0, 0), width=2.4, height=2.4, theta1=3*90+t1, theta2=3*90+t2, linewidth=2, zorder=3)

a5 = Arc(xy=(0, 0), width=0.9, height=0.9, theta1=0*90+t1, theta2=0*90+t2, linewidth=2, zorder=3)
a6 = Arc(xy=(0, 0), width=0.5, height=0.5, theta1=t2, theta2=t2+t1, linewidth=2, zorder=3)
a7 = Arc(xy=(0, 0), width=0.5, height=0.5, theta1=0, theta2=t1, linewidth=2, zorder=3)


ax = plt.gca()
ax.set_axis_off()
ax.add_patch(r)
ax.add_patch(c)
ax.add_line(lh)
ax.add_line(lv)
ax.add_line(l4)
ax.add_line(l5)
ax.add_patch(a1)
ax.add_patch(a2)
ax.add_patch(a3)
ax.add_patch(a4)
ax.add_patch(a5)
ax.add_patch(a6)
ax.add_patch(a7)


ax.text(0.35, 0.35, r'$\beta$', fontsize=24)
ax.text(0.05, 0.3, r'$\alpha$', fontsize=24)
ax.text(0.3, 0.05, r'$\alpha$', fontsize=24)

ax.text(0.5, -0.14, '$L/2$', fontsize=24, horizontalalignment='center')


plt.ylim(-1.35, 1.35)
plt.xlim(-1.35, 1.35)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()


filename_base = __file__.split('.')[0]
plt.savefig(filename_base + '.pdf', bbox_inces='tight', pad_inches=0)
plt.show(block=True)


