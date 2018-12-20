import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt
from matplotlib import rc

plt.figure(figsize=(7, 5))

#rc('axes', linewidth = 2)

seed = 34
np.random.seed(seed)
plt.cla()
sample = np.random.uniform(0, 1, 100)
sample.sort()

y = [i+1 for i in range(len(sample))]
y = np.array(y)/100.

plt.step(sample, y, label='$S_n(x)$', linewidth=2)
plt.xlim = (0, 1)

xs = np.linspace(0, 1, 100)
ys = xs
plt.plot(xs, ys, '-g', label='$F_0(x)$', linewidth=2)

diff = list(np.abs(y-sample))
i = diff.index(max(diff))
x = sample[i]

plt.vlines(x, min(y[i], sample[i]), max(y[i], sample[i]), color='red', zorder=10, linewidth=2)
plt.legend(loc='best')
plt.xlabel('$x$', fontsize=17)
plt.ylabel('$S_n(x), \; F_0(x)$', fontsize=17)

fontprop = {'size': 13}
a = plt.gca()
rc('font',**fontprop)
a.set_xticklabels(a.get_xticks(), fontprop)
a.set_yticklabels(a.get_yticks(), fontprop)

print min(y[i], sample[i]) - max(y[i], sample[i])
print s.kstest(sample, 'uniform', alternative='two_sided')

#plt.show()
plt.savefig('edf' + '.pdf', bbox_inces='tight')

