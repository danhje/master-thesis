import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D                        


fig = plt.figure(figsize=(6,6), frameon=False)

t1 = np.arccos(1/1.2)*360/(2*np.pi)
t2 = 90 - t1

r = Rectangle((-1, -1), width=2, height=2, fill=False, edgecolor='black', linewidth=1, zorder=5)
c = Circle((0, 0), radius=1.2, facecolor='none', linestyle='solid', edgecolor='black', linewidth=1, zorder=2)
lh = Line2D([-1.35, 1.35], [0, 0], color='black', linestyle='--', linewidth=1, zorder=0)
lv = Line2D([0, 0], [-1.35, 1.35], color='black', linestyle='--', linewidth=1, zorder=0)

s = 22.8
gc1 = Arc(xy=(0, -0.5), width=2*(1.2+0.5), height=2*(1.2+0.5)-0.143, theta1=90, theta2=90+s, linewidth=2, zorder=3, color='blue')
gc2 = Arc(xy=(0, -0.5), width=2*(1.2+0.5), height=2*(1.2+0.5)-0.143, theta1=90-s, theta2=90, linewidth=2, zorder=3)

a = Line2D([0, 0], [0, 1.128], color='blue', linestyle='-', linewidth=2, zorder=0)

y = 1; x = np.tan(np.arccos(1/1.2))
lP = Line2D([0, -x], [0, y], color='blue', linestyle='-', linewidth=2, zorder=3)
lQ = Line2D([0, x], [0, y], color='black', linestyle='-', linewidth=2, zorder=3)

gamma = Arc(xy=(0, 0), width=0.5, height=0.5, theta1=90, theta2=90+t1, linewidth=2, zorder=3)
alpha = Arc(xy=(-x, y), width=0.3, height=0.3, theta1=-56, theta2=16, linewidth=2, zorder=3)
beta_h = Line2D([-0.15, 0], [0.98, 0.98], color='black', linestyle='-', linewidth=2, zorder=0)
beta_v = Line2D([-0.15, -0.15], [0.98, 1.12], color='black', linestyle='-', linewidth=2, zorder=0)

ax = plt.gca()
ax.set_axis_off()
ax.add_patch(r)
ax.add_patch(c)
ax.add_patch(gc1)
ax.add_patch(gc2)
ax.add_line(lh)
ax.add_line(lv)
ax.add_line(lP)
ax.add_line(lQ)
ax.add_line(a)
ax.add_patch(gamma)
ax.add_patch(alpha)
ax.add_line(beta_h)
ax.add_line(beta_v)


ax.text(-0.11, 0.29, r'$\gamma$', fontsize=20, horizontalalignment='center', verticalalignment='center')
ax.text(-x+0.2, y-0.08, r'$\alpha$', fontsize=20, horizontalalignment='center', verticalalignment='center')
ax.text(0-0.22, y-0.08, r'$\beta$', fontsize=20, horizontalalignment='center', verticalalignment='center')

ax.text(0.07, 0.6, r'$a$', fontsize=20, horizontalalignment='center', verticalalignment='center')
ax.text(-0.45, 0.5, r'$b$', fontsize=20, horizontalalignment='center', verticalalignment='center')
ax.text(-0.31, 1.05, r'$c$', fontsize=20, horizontalalignment='center', verticalalignment='center')
ax.text(0.3, 1.3, r'$\frac{1}{2}E$', fontsize=20, horizontalalignment='center', verticalalignment='center')
ax.arrow(0.2, 1.3, -0.15, -0.13, shape='full', head_width=0.05, head_length=0.07, fc="k", ec="k", length_includes_head="True")
ax.text(-0.2, 0.6, r'$T$', fontsize=20, horizontalalignment='center', verticalalignment='center')

ax.text(-0.72, 1.03, '$\mathrm{P}$', fontsize=20, horizontalalignment='center')
ax.text(0.72, 1.03, '$\mathrm{Q}$', fontsize=20, horizontalalignment='center')
ax.text(0.1, -0.15, '$\mathrm{R}$', fontsize=20, horizontalalignment='center')


plt.ylim(-1.35, 1.35)
plt.xlim(-1.35, 1.35)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()


filename_base = __file__.split('.')[0]
plt.savefig(filename_base + '.pdf', bbox_inces='tight', pad_inches=0)
plt.show(block=True)
plt.cla()
plt.close('all')

