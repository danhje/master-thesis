import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

r = 0.7

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z,  rstride=4, cstride=4, linewidth=1, color='b', alpha=0.5, zorder=0)

l = 0.5

# Face 1
x1 = np.array([[-l, l, l, -l], 
               [-l, -l, -l, -l]]) 
y1 = np.array([[-l, -l, -l, -l], 
               [-l, -l, -l, -l]]) 
z1 = np.array([[-l, -l, l, l], 
               [-l, -l, -l, -l]]) 
# Face 2
x2 = np.array([[-l, -l, -l, -l], 
               [-l, -l, -l, -l]]) 
y2 = np.array([[-l, l, l, -l], 
               [-l, -l, -l, -l]]) 
z2 = np.array([[-l, -l, l, l], 
               [-l, -l, -l, -l]]) 
# Face 3 
x3 = np.array([[-l, l, l, -l], 
               [-l, -l, -l, -l]]) 
y3 = np.array([[-l, -l, l, l], 
               [-l, -l, -l, -l]]) 
z3 = np.array([[l, l, l, l], 
               [l, l, l, l]]) 
# Face 4 
x4 = np.array([[-l, l, l, -l], 
               [-l, -l, -l, -l]]) 
y4 = np.array([[l, l, l, l], 
               [l, l, l, l]]) 
z4 = np.array([[-l, -l, l, l], 
               [-l, -l, -l, -l]]) 
# Face 5 
x5 = np.array([[-l, -l, l, l], 
               [-l, -l, -l, -l]]) 
y5 = np.array([[-l, l, l, -l], 
               [-l, -l, -l, -l]]) 
z5 = np.array([[-l, -l, -l, -l], 
               [-l, -l, -l, -l]]) 
# Face 6 
x6 = np.array([[l, l, l, l], 
               [l, l, l, l]]) 
y6 = np.array([[-l, l, l, -l], 
               [-l, -l, -l, -l]]) 
z6 = np.array([[-l, -l, l, l], 
               [-l, -l, -l, -l]]) 

ax.plot_surface(x1,y1,z1, color='purple', alpha=1, zorder=10)
ax.plot_surface(x2,y2,z2, color='purple', alpha=1, zorder=10)
ax.plot_surface(x3,y3,z3, color='purple', alpha=1, zorder=10)
ax.plot_surface(x4,y4,z4, color='purple', alpha=1, zorder=10)
ax.plot_surface(x5,y5,z5, color='purple', alpha=1, zorder=10)
ax.plot_surface(x6,y6,z6, color='purple', alpha=1, zorder=10)






plt.show(block=True)

