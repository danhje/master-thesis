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
    Class used for testing RandomConvergentConnect.
    '''

    def __init__(self, N_s, N_t, C, e_min=10, threads=1):
        '''
        Sets up the experiment, and calculates expected distributions
        for later comparison with the observed distribution.

        Parameters
        ----------
            N_s    : Number of source neurons.
            N_t    : Number of target neurons.
            C      : In-degree (number of connections per 
                     target neuron).
            e_min  : Minimum expected number of observations in
                     each bin. Default is 10.
            threads: Set number of local threads. Default is 1.
        '''

        self.N_s = N_s
        self.N_t = N_t
        self.C = C
        self.e_min = e_min
        self.threads = threads

        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': self.threads})
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')

        expected_degree = self.N_t * self.C / float(self.N_s)
        if expected_degree < self.e_min:
            raise RuntimeWarning(
                'Expected out-degree (%.2f) is less than e_min (%.2f).' \
                'Increase N_t*C / N_s or decrease e_min.' % \
                (expected_degree, self.e_min))

        self.expected = [expected_degree] * self.N_s

    def _counter(self, x):
        '''
        Count similar elements in list.
        
        Parameters
        ----------
            x: Any list.
            
        Return values
        -------------
            list containing counts of similar elements.
        '''

        x.sort()
        counts = []
        last_elem = None
        for elem in x:
            if elem == last_elem:
                counts[-1] += 1
            else:
                counts.append(1)
            last_elem = elem

        # Append 0s if necessary.
        length = len(counts)
        if length < self.N_s:
            counts.extend([0] * (self.N_s - length))

        return counts

    def _reset(self, msd):
        '''
        Reset the NEST kernel and set seed values.

        Parameters
        ----------
            msd: Master RNG seed.
        '''

        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': self.threads})
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')

        # Set PRNG seed values:
        if msd == None:
            msd = rnd.randint(1000000)
        msdrange = range(msd, msd + self.n_vp)
        nest.SetKernelStatus({'grng_seed': msd + self.n_vp,
                              'rng_seeds': msdrange})

    def _build(self):
        '''Create all nodes.'''

        self.source_nodes = nest.Create('iaf_neuron', self.N_s)
        self.target_nodes = nest.Create('iaf_neuron', self.N_t)

    def _connect(self):
        '''Connect all nodes.'''

        nest.RandomConvergentConnect(self.source_nodes, self.target_nodes,
                                     self.C, options={'allow_multapses': True})

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
        source_connections = [conn[0] for conn in connections]
        degrees = self._counter(source_connections)

        return degrees

    def _get_degrees_control(self, msd):
        '''
        Instead of using NEST, this method returns data with the expected
        multinomial distribution. 
        
        Parameters
        ----------
            msd: Master RNG seed.
        
        Return values
        -------------
            list containing the out-degrees of the source nodes.
        '''

        if msd != None:
            rnd.seed(msd)

        con = rnd.randint(0, self.N_s, self.N_t * self.C)
        degrees = self._counter(con)

        return degrees

    def chi_squared_test(self, msd=None, control=False):
        '''
        Create a single network and compare the resulting out-degree
        distribution with the expected distribution using Pearson's chi-squared
        GOF test.

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

        # ddof: adjustment to the degrees of freedom. df = k-1-ddof
        return scipy.stats.chisquare(numpy.array(degrees),
                                     numpy.array(self.expected), ddof=0)

    def two_level_test(self, n_runs, start_seed=None, control=False,
                       show_histogram=False, histogram_bins=100,
                       show_CDF=False):
        '''
        Create a network and run chi-squared GOF test n_runs times.
        Test whether resulting p-values are uniformly distributed
        on [0, 1] using the Kolmogorov-Smirnov GOF test.

        Parameters
        ----------
            n_runs        : number of times to repeat chi-squared test.
            start_seed    : First PRNG seed value.
            control       : boolean value. If True, _get_degrees_control will be
                            used instead of _get_degrees.
            show_histogram: Specify whether histogram should be displayed.
            histogram_bins: Number of histogram bins.
            show_CDF      : Specify whether EDF should be displayed.

        Return values
        -------------
            KS statistic.
            p-value from KS test.
        '''

        self.pvalues = []

        if start_seed == None:
            for i in range(n_runs):
                print 'Running test %d of %d.' % (i + 1, n_runs)
                chi, p = self.chi_squared_test(None, control)
                self.pvalues.append(p)
        else:
            seed_jump = self.n_vp + 1
            end_seed = start_seed + n_runs * seed_jump
            for seed in range(start_seed, end_seed, seed_jump):
                print 'Running test %d of %d.' % \
                    (1 + (seed - start_seed) / seed_jump, n_runs)
                chi, p = self.chi_squared_test(seed, control)
                self.pvalues.append(p)

        ks, p = scipy.stats.kstest(self.pvalues, 'uniform',
                                   alternative='two_sided')

        if show_CDF:
            plt.figure()
            self.pvalues.sort()
            y = [(i + 1.) / len(self.pvalues)
                 for i in range(len(self.pvalues))]
            plt.step([0.0] + self.pvalues + [1.0], [0.0] + y + [1.0])
            plt.xlabel('P-values')
            plt.ylabel('Empirical distribution function')

        if show_histogram:
            plt.figure()
            plt.hist(self.pvalues, bins=histogram_bins)
            plt.xlabel('P-values')
            plt.ylabel('Frequency')

        if show_CDF or show_histogram:
            plt.show(block=True)

        return ks, p


if __name__ == '__main__':
    test = RCC_tester(N_s=1000, N_t=1000, C=1000)
    ks, p = test.two_level_test(n_runs=1000, start_seed=0)
    print 'KS test statistic:', ks
    print 'p-value of KS-test of uniformity:', p

