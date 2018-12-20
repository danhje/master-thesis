'''
@author: Daniel Hjertholm

Tests for RandomConvergentConnect.
'''

import numpy
import numpy.random as rnd
import scipy.stats
import matplotlib.pyplot as plt
import nest


class RCC_tester(object):
    '''
    Experiment class for testing RandomConvergentConnect.
    '''

    def __init__(self, N_s, N, C, e_min=10):
        '''
        Initialize an Experiment object.

        Sets up the experiment, and calculates expected distributions
        for later comparison with the observed distribution.

        Parameters
        ----------
            N_s  : Number of source neurons.
            N  : Number of target neurons.
            C    : In-degree (number of connections per 
                   target neuron)
            e_min: Minimum expected number of observations in
                   each bin.
        '''

        self.N_s = N_s
        self.N = N
        self.C = C
        self.e_min = e_min

        nest.ResetKernel()
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')

        expected_degree = self.N * self.C / float(self.N_s)
        if expected_degree < self.e_min:
            raise RuntimeWarning(
                'Expected out-degree (%.2f) is less than e_min' \
                '(%.2f). Increase N*C / N_s or decrease e_min.' %
                (expected_degree, self.e_min))

        self.expected = [expected_degree] * self.N_s

    def _counter(self, x):
        '''
        Count elements in iterable.
        
        Parameters
        ----------
            x: List, array or other iterable.
            
        Return values
        -------------
            dict containing key-value pairs of elements from x and 
            their counts.
        '''

        counts = {}
        for elem in x:
            if elem not in counts:
                counts[elem] = 1
            else:
                counts[elem] += 1

        return counts

    def _reset(self, msd):
        '''
        Reset the NEST kernel and set seed values.

        Parameters
        ----------
            msd: Master RNG seed.
        '''

        nest.ResetKernel()
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')

        # Set PRNG seed values:
        if msd == None:
            msd = rnd.randint(1000000)
        msdrange = range(msd, msd + self.n_vp)
        nest.SetKernelStatus({'grng_seed': msd + self.n_vp,
                              'rng_seeds': msdrange})

    def _build(self):
        '''Create all nodes.'''

        self.source_nodes = nest.Create("iaf_neuron", self.N_s)
        self.target_nodes = nest.Create("iaf_neuron", self.N)

    def _connect(self):
        '''Connect all nodes.'''

        nest.RandomConvergentConnect(self.source_nodes,
            self.target_nodes,
            self.C,
            options={'allow_multapses': True})

    def _get_degrees(self, msd):
        '''
        Reset NEST, create nodes, connect them, and retrieve the 
        resulting connections.
        
        Parameters
        ----------
            msd: Master RNG seed.
        
        Return values
        -------------
            list containing the out-degrees of the source nodes.
        '''

        self._reset(msd)
        self._build()
        self._connect()

        connections = nest.GetConnections(source=self.source_nodes)
        connections = numpy.array(connections)[:, 0]
        degrees = self._counter(connections).values()

        return degrees

    def _get_degrees_control(self, msd):
        '''
        Instead of using NEST, this method creates a "fake" degree
        list, using an algorithm similar to the one NEST supposedly
        uses, with pseudorandom numbers from numpy.random.randint().
        
        Parameters
        ----------
            msd: Master RNG seed.
        
        Return values
        -------------
            list containing the out-degrees of the source nodes.
        '''

        if msd != None:
            rnd.seed(msd)

        con = rnd.randint(0, self.N_s, self.N * self.C)
        degrees = self._counter(con).values()

        # Bias 1
        #'''
        length = len(degrees)
        for i in range(length / 2):
            degrees[i] -= 1
            degrees[length / 2 + i] += 1
        #'''

        # Bias 2
        '''
        length = len(degrees)
        for i in range(0, length, 2):
            degrees[i] -= 1
            degrees[i + 1] += 1
        '''

        # Bias 3
        '''
        degrees[0] += 1
        degrees[-1] -= 1
        '''

        # Bias 4
        '''
        num = int(self.N_s * 0.25)
        c = zip(degrees, range(len(degrees)))
        c.sort()
        for deg, ind in c[0:25]:
            degrees[ind] -= 1
        for deg, ind in c[-25:]:
            degrees[ind] += 1
        '''

        # Bias 5
        '''
        num = int(self.N_s * 0.05)
        c = zip(degrees, range(len(degrees)))
        c.sort()
        for deg, ind in c[0:num]:
            degrees[ind] += 1
        for deg, ind in c[-1 * num:]:
            degrees[ind] -= 1
        '''

        # Bias 6
        '''
        num = int(self.N_s * 0.01)
        c = zip(degrees, range(len(degrees)))
        c.sort()
        for deg, ind in c[0:num]:
            degrees[ind] += 1
        for deg, ind in c[-1 * num:]:
            degrees[ind] -= 1
        '''

        # Bias 7
        '''
        degrees[-1] += degrees[0]
        degrees[0] = 0
        '''

        if not sum(degrees) == self.N * self.C:
            raise RuntimeWarning("Something's wrong.")

        return degrees

    def chi_squared_test(self, msd=None, control=False):
        '''
        Create a single network and compare the resulting out-degree
        distribution with the expected distribution using Pearson's
        chi-squared GOF test.

        Parameters
        ----------
            msd    : Master RNG seed.
            control: Boolean value. If True, _get_degrees_control will
                     be used instead of _get_degrees.

        Return values
        -------------
            chi-squared statistic.
            p-value from chi-squared test.
        '''

        if control:
            degrees = self._get_degrees_control(msd)
        else:
            degrees = self._get_degrees(msd)

        # If some source nodes have an out-degree of 0, then these 
        # will not be in degrees. They must be added manually.
        length = len(degrees)
        if length < self.N_s:
            degrees.extend([0.0] * (self.N_s - length))

        # ddof: adjustment to the degrees of freedom. df = k-1-ddof
        return scipy.stats.chisquare(numpy.array(degrees),
                                     numpy.array(self.expected),
                                     ddof=0)

    def ks_test(self, n, start_seed=None, control=False,
                show_histogram=False, histogram_bins=100,
                show_CDF=False):
        '''
        Create a network and run chi-squared GOF test n times.
        Test whether resulting p-values are uniformly distributed
        on [0, 1] using the Kolmogorov-Smirnov GOF test.

        Parameters
        ----------
            n             : number of times to repeat chi-squared
                            test.
            start_seed    : First PRNG seed value.
            control       : boolean value. If True, 
                            _get_degrees_control will be used instead 
                            of _get_degrees.
            show_histogram: Specify whether histogram should
                            be displayed.
            histogram_bins: Specify the number of histogram bins.
            show_CDF      : Specify whether CDF should be displayed.

        Return values
        -------------
            KS statistic.
            p-value from KS test.
        '''

        self.pvalues = []

        if start_seed == None:
            for i in range(n):
                print 'Running test %d of %d.' % \
                    (i + 1, n)
                chi, p = self.chi_squared_test(msd=None,
                                             control=control)
                self.pvalues.append(p)
        else:
            seed_jump = self.n_vp + 1
            seed_end = start_seed + n * seed_jump
            for seed in range(start_seed, seed_end, seed_jump):
                print 'Running test %d of %d.' % \
                    (1 + (seed - start_seed) / seed_jump, n)
                chi, p = self.chi_squared_test(seed, control)
                self.pvalues.append(p)

        ks, p = scipy.stats.kstest(self.pvalues, 'uniform',
                                   alternative='two_sided')

        filename_base = __file__.split('.')[0]

        if show_CDF:
            x = numpy.linspace(0, 1, 100)
            y = x
            plt.plot(x, y, '-', color='grey', linewidth=10, label='$\mathrm{Theory}$')
            self.pvalues.sort()
            y = [(i + 1.) / len(self.pvalues)
                 for i in range(len(self.pvalues))]
            plt.step([0.0] + self.pvalues + [1.0], [0.0] + y + [1.0], color='red', 
                     linewidth=3,
                     label='$\mathrm{Observation}$')
            
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.xlabel('$x$', fontsize=28)
            plt.ylabel('$F_0(x), \, S_n(x)$', fontsize=28)

            fontprop = {'size': 20}
            a = plt.gca()
            from matplotlib import rc
            rc('font', **fontprop)
            a.set_xticklabels(a.get_xticks(), fontprop)
            a.set_yticklabels(a.get_yticks(), fontprop)

            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(filename_base + '_CDF.pdf', bbox_inces='tight')
            plt.cla()
            plt.close('all')

        if show_histogram:
            if show_CDF:
                plt.figure()

            x = [0,1]
            y = [100, 100]
            plt.plot(x, y, linewidth=10, color='grey', label='$\mathrm{Theory}$')

            b = plt.hist(self.pvalues, bins=histogram_bins, histtype='step', color='red', linewidth=3, zorder=9)
            import numpy as np
            p=b[2][0]
            xy=p.get_xy()
            xy[0,0]=np.nan
            xy[-1,0]=np.nan
            p.set_xy(xy)
            
            plt.plot((-1), (-1), label='$\mathrm{Observation}$', linewidth=3, color='red')
            
            plt.xlim((0, 1))
            plt.ylim(ymin=0)
            plt.xlabel('$\mathrm{p-values}$', fontsize=28)
            plt.ylabel('$\mathrm{Frequency}$', fontsize=28)
            plt.legend(loc='lower right')
        
            plt.tight_layout()
            plt.savefig(filename_base + '_hist.pdf', bbox_inces='tight')
            plt.close('all')

        return ks, p, a


if __name__ == "__main__":
    test = RCC_tester(N_s=1000, N=1000, C=100)
    ks, p, a = test.ks_test(n=10000, start_seed=0, control=True, show_CDF=True, show_histogram=True)
    print 'KS test statistic:', ks
    print 'p-value of KS-test of uniformity:', p

