import numpy as np
from mayavi import mlab

fig = mlab.figure(bgcolor=(1,1,1), size=(700, 700))

c = (0.4,0.4,0.4)

x = np.array((-1, 1))
y = np.array((-1, 1))
s = np.array(((1,1),(1,1)))
s1 = mlab.surf(x, y, s, color=c, opacity=0.7)
s2 = mlab.surf(x, y, -s, color=c, opacity=0.7)

x = np.array((-1, -1))
y = np.array((-1, 1))
s = np.array(((-1,-1),(1,1)))
s3 = mlab.surf(x, y, s, color=c, opacity=0.7)
s4 = mlab.surf(-x, y, s, color=c, opacity=0.7)

x = np.array((-1, 1))
y = np.array((1, 1))
s = np.array(((-1,1),(-1,1)))
s5 = mlab.surf(x, y, s, color=c, opacity=0.7)
s6 = mlab.surf(x, -y, s, color=c, opacity=0.7)

r1 = 0.9
r2 = 1.3
r3 = 1.6

r = r3

pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2*pi:101j]

x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

mlab.mesh(x, y, z, color=(0.3, 0.3, 1), opacity=1)

d = np.sqrt(r**2 - 1.0**2)
x = np.linspace(-d, d, 100)
y = np.sqrt(d**2 - x**2)
z = np.array([1.01]*100)
x = np.concatenate((x[:-1],x[::-1]))
y = np.concatenate((y[:-1],(y*-1)[::-1]))
z = np.concatenate((z[:-1], z[::-1]))
'''
mlab.plot3d(x, y, z, tube_radius=None, line_width=5, color=(0., 0., 0.))
mlab.plot3d(z, y, x, tube_radius=None, line_width=5, color=(0., 0., 0.))
mlab.plot3d(x, z, y, tube_radius=None, line_width=5, color=(0., 0., 0.))
'''

mlab.view(distance=7)
mlab.show()

