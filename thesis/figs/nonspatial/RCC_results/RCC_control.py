import numpy
import numpy.random as rnd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rc
import os


os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

filename = 'RCC_9_pvs.txt'
pvalues = numpy.loadtxt(filename)
pvalues = list(pvalues)
print 'Removing p-value', pvalues.pop(-1) # The file contains 101 pvalues by mistake. Removing the last one.
pvalues.sort()
print pvalues[-1]


filename_base = __file__.split('.')[0]


ks2, p2 = scipy.stats.kstest(pvalues, 'uniform', alternative='two_sided')

f = open(filename_base + '_res.txt', 'a')
f.write(str(ks2) + ', ' + str(p2) + '\n')
f.close()


# Create EDF plot.
#'''
figEDF = plt.figure(1, figsize=(6.0, 5.0))

xs = numpy.linspace(0, 1, 100)
ys = xs
plt.plot(xs, ys, label='$\mathrm{Theory}$', color='grey', linewidth=5)

y = [(o + 0.) / len(pvalues)
     for o in range(len(pvalues))]
plt.step([0.0] + pvalues + [1.0], [0.0] + y + [1.0], label='$\mathrm{Observation}$', zorder=10, color='red', linewidth=1.5)

plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$F_0(x) \, S_n(x)$', fontsize=13)
fontprop = {'size': 10}
a = plt.gca()
rc('font', **fontprop)
a.set_xticklabels(a.get_xticks(), fontprop)
a.set_yticklabels(a.get_yticks(), fontprop)
plt.legend(loc='upper left')
plt.savefig(filename_base + '_EDF.pdf', bbox_inches='tight')
plt.cla()
plt.close('all')
#'''


# Create histogram.
'''
plt.figure(2, figsize=(5, 4))
plt.hist(pvalues, bins=100)
rc('font',**{'family':'serif','serif':['cmr10']})
rc('text', usetex=True)
plt.xlabel(r'\text{p-values}', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
#fontprop = {'size': 10}
a = plt.gca()
#rc('font', **fontprop)
#a.set_xticklabels(a.get_xticks(), fontprop)
#a.set_yticklabels(a.get_yticks(), fontprop)
plt.tight_layout()
plt.savefig(filename_base + '_hist.pdf', bbox_inces='tight')
plt.close('all')
'''

