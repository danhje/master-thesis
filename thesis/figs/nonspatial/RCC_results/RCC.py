import numpy
import numpy.random as rnd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rc
import os


os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

filename = 'RCC_8_pvs.txt'
pvalues = numpy.loadtxt(filename)
pvalues = list(pvalues)
pvalues.sort()


# Create EDF plot.
#'''
plt.figure(1, figsize=(6.0, 5.0))

xs = numpy.linspace(0, 1, 100)
ys = xs
plt.plot(xs, ys, '-g', label='$\mathrm{Theory}$', color='grey', linewidth=10, zorder=9)


y = [(o + 1.) / len(pvalues)
     for o in range(len(pvalues))]
plt.step([0.0] + pvalues + [1.0], [0.0] + y + [1.0], label='$\mathrm{Observation}$', zorder=10, linewidth=3, color='red')

plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('$\mathrm{p-values}$', fontsize=28)
plt.ylabel('$\mathrm{Frequency}$', fontsize=28)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('RCC_EDF.pdf', pad_inches=0)
plt.cla()
plt.close('all')
#'''


# Create histogram.
#'''
plt.figure(2, figsize=(6.0, 5.0))
plt.hist(pvalues, bins=100)
plt.xlabel('$\mathrm{p-values}$', fontsize=28)
plt.ylabel('$\mathrm{Frequency}$', fontsize=28)

plt.tight_layout()
plt.savefig('RCC_hist.pdf', pad_inches=0)
plt.cla()
plt.close('all')
#'''

