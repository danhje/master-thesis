'''
Created January 2013
@author: Daniel Hjertholm

Tests for RandomConvergentConnect.
'''

import nest
import numpy
import numpy.random as rnd
import scipy.stats
import matplotlib.pyplot as plt


class RCC_tester(object):
    '''
    Experiment class for testing RandomConvergentConnect.
    '''

    def __init__(self, N_s, N_t, C, e_min=10):
        '''
        Initialize an Experiment object.

        Sets up the experiment, and calculates expected distributions
        for later comparison with the observed distribution.

        Parameters
        ----------
            N_s  : Number of source neurons.
            N_t  : Number of target neurons.
            C    : In-degree (number of connections per 
                   target neuron)
            e_min: Minimum expected number of observations in
                   each bin.
        '''

        self.N_s = N_s
        self.N_t = N_t
        self.C = C
        self.e_min = e_min

        nest.ResetKernel()
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')

        expected_degree = self.N_t * self.C / float(self.N_s)
        if expected_degree < self.e_min:
            raise RuntimeWarning(
                'Expected out-degree (%.2f) is less than e_min' \
                '(%.2f). Increase N_t*C / N_s or decrease e_min.' %
                (expected_degree, self.e_min))

        self.expected = [expected_degree] * self.N_s

    def _counter(self, x):
        '''
        Count elements in iterable.
        
        Parameters
        ----------
            x: list, array or other iterable.
            
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
            msd: master RNG seed.
        '''

        nest.ResetKernel()

        # Set PRNG seed values:
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        msdrange = range(msd, msd + self.n_vp)
        nest.SetKernelStatus({'grng_seed': msd + self.n_vp,
                              'rng_seeds': msdrange})

    def _build(self):
        '''
        Create all nodes.
        '''

        self.source_nodes = nest.Create("iaf_neuron", self.N_s)
        self.target_nodes = nest.Create("iaf_neuron", self.N_t)

    def _connect(self):
        '''
        Connect all nodes.
        '''

        nest.RandomConvergentConnect(self.source_nodes,
            self.target_nodes,
            self.C,
            options={'allow_multapses': True})

    def _get_degrees(self, msd):
        '''
        Reset NEST, create nodes, connect them, and retrieve the 
        resulting connections.
        
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
        
        Return values
        -------------
            list containing the out-degrees of the source nodes.
        '''

        rnd.seed(msd)
        con = rnd.randint(0, self.N_s, self.N_t * self.C)
        degrees = self._counter(con).values()

        return degrees

    def chi_squared_test(self, msd):
        '''
        Create a single network and compare the resulting out-degree
        distribution with the expected distribution using Pearson's
        chi-squared GOF test.

        Parameters
        ----------
            msd: master RNG seed.

        Return values
        -------------
            chi-squared statistic.
            p-value from chi-squared test.
        '''

        degrees = self._get_degrees(msd)
        #degrees = self._get_degrees_control(msd)

        # If some source nodes have an out-degree of 0, then these 
        # will not be in self.degrees. They must be added manually.
        length = len(degrees)
        if length < self.N_s:
            degrees.extend([0.0] * (self.N_s - length))

        # ddof: adjustment to the degrees of freedom. df = k-1-ddof
        return scipy.stats.chisquare(numpy.array(degrees),
                                     numpy.array(self.expected),
                                     ddof=0)

    def ks_test(self, n, seed_start=0,
                show_histogram=False, histogram_bins=10,
                show_CDF=False):
        '''
        Create a network and run chi-squared GOF test n times.
        Test whether resulting p-values are uniformly distributed
        on [0, 1] using the Kolmogorov-Smirnov GOF test.

        Parameters
        ----------
            n             : number of times to repeat chi-squared
                            test.
            seed_start    : First PRNG seed value.
            show_histogram: Specify whether histogram should
                            be displayed.
            histogram_bins: Specify the number of histogram bins.

        Return values
        -------------
            KS statistic.
            p-value from KS test.
        '''

        seed_jump = self.n_vp + 1
        seed_end = seed_start + n * seed_jump

        self.pvalues = []

        for seed in range(seed_start, seed_end, seed_jump):
            print 'Running test %d of %d.' % \
                (1 + (seed - seed_start) / seed_jump, n)
            chi, p = self.chi_squared_test(seed)
            self.pvalues.append(p)

        ks, p = scipy.stats.kstest(self.pvalues, 'uniform',
                                   alternative='two_sided')

        if show_CDF:
            self.pvalues.sort()
            y = [(i + 1.) / len(self.pvalues)
                 for i in range(len(self.pvalues))]
            plt.step([0.0] + self.pvalues + [1.0], [0.0] + y + [1.0])
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.draw()
            raw_input('Press enter to continue... ')

        if show_histogram:
            if show_CDF:
                plt.figure()
            plt.hist(self.pvalues, bins=histogram_bins)
            plt.draw()
            raw_input('Press enter to continue... ')

        return ks, p


if __name__ == "__main__":
    e = RCC_tester(N_s=1000, N_t=1000, C=1000, e_min=10)
    ks, p = e.ks_test(n=10000, seed_start=0)
    print 'KS test statistic:', ks
    print 'p-value of KS-test of uniformity:', p

    filename_base = __file__.split('.')[0]

    numpy.savetxt(filename_base + '_pvs.txt', e.pvalues)

    f = open(filename_base + '_res.txt', 'a')
    f.write(str(ks) + ', ' + str(p) + '\n')
    f.close()

    # Save plot of EDF.
    #'''
    plt.figure(figsize=(5, 4))
    e.pvalues.sort()
    y = [(i + 1.) / len(e.pvalues)
         for i in range(len(e.pvalues))]
    plt.step([0.0] + e.pvalues + [1.0], [0.0] + y + [1.0], label='$S_n(x)$')
    xs = numpy.linspace(0, 1, 100)
    ys = xs
    plt.plot(xs, ys, '-g', label='$F_0(x)$')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$S_n(x), \; F_0(x)$', fontsize=13)
    fontprop = {'size': 10}
    a = plt.gca()
    from matplotlib import rc
    rc('font', **fontprop)
    a.set_xticklabels(a.get_xticks(), fontprop)
    a.set_yticklabels(a.get_yticks(), fontprop)
    plt.savefig(filename_base + '_EDF.pdf', bbox_inces='tight')
    plt.close('all')
    #'''

    # Save histogram.
    #'''
    plt.figure(figsize=(5, 4))
    plt.hist(e.pvalues, bins=100)
    fontprop = {'size': 10}
    a = plt.gca()
    from matplotlib import rc
    rc('font', **fontprop)
    a.set_xticklabels(a.get_xticks(), fontprop)
    a.set_yticklabels(a.get_yticks(), fontprop)
    plt.savefig(filename_base + '_hist.pdf', bbox_inces='tight')
    plt.close('all')
    #'''

