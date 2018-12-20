f = open('2D_exponential_rerun_pvalues.txt', 'r')
data = f.readlines()
f.close

ps = []
for line in data:
    n, msd, ks, p = line.split(',')
    ps.append(float(p))

import scipy.stats as s
ks, p = s.kstest(ps, 'uniform', alternative='two_sided')
print ks, p

print sum([1 for p in ps if p<0.01])

#import matplotlib.pyplot as plt
#plt.hist(ps)
#plt.show(block=True)
