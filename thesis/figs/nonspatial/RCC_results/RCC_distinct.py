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

    def __init__(self, N_s, N_t, C, e_min=10, threads=1):
        '''
        Initialize an Experiment object.

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
        nest.SetKernelStatus({"local_num_threads": self.threads})
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
        nest.SetKernelStatus({"local_num_threads": self.threads})
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
        self.target_nodes = nest.Create("iaf_neuron", self.N_t)

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

        con = rnd.randint(0, self.N_s, self.N_t * self.C)
        degrees = self._counter(con).values()

        return degrees

    def chiSquaredTest(self, msd=None, control=False):
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
                chi, p = self.chiSquaredTest(msd=None,
                                             control=control)
                self.pvalues.append(p)
        else:
            seed_jump = self.n_vp + 1
            seed_end = start_seed + n * seed_jump
            for seed in range(start_seed, seed_end, seed_jump):
                print 'Running test %d of %d.' % \
                    (1 + (seed - start_seed) / seed_jump, n)
                chi, p = self.chiSquaredTest(seed, control)
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
            plt.show()
            raw_input('Press enter to continue... ')

        if show_histogram:
            if show_CDF:
                plt.figure()
            plt.hist(self.pvalues, bins=histogram_bins)
            plt.show()
            raw_input('Press enter to continue... ')

        return ks, p


if __name__ == "__main__":
    from matplotlib import rc
    f = open('RCC_distinct.txt', 'a')

    
    plt.figure()
    test = RCC_tester(N_s=10, N_t=5, C=5, e_min=0.01)
    ks, p = test.ks_test(n=10000, start_seed=0, control=True)
    print 'p-value of KS-test of uniformity:', p
    f.write(str(ks) + ', ' + str(p) + '\n')
    test.pvalues.sort()
    y = [(i + 1.) / len(test.pvalues) for i in range(len(test.pvalues))]
    plt.step([0.0] + test.pvalues + [1.0], [0.0] + y + [1.0], color='blue')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('$x$', fontsize=32)
    plt.ylabel('$S_n(x)$', fontsize=32)
    fontprop = {'size': 20}
    a = plt.gca()
    rc('font', **fontprop)
    a.set_xticklabels(a.get_xticks(), fontprop)
    a.set_yticklabels(a.get_yticks(), fontprop)
    plt.tight_layout()
    plt.savefig('RCC_distinct_A.pdf')


    plt.figure()
    test = RCC_tester(N_s=10, N_t=10, C=5, e_min=0.01)
    # Have to change e_min!
    ks, p = test.ks_test(n=10000, start_seed=0, control=True)
    print 'p-value of KS-test of uniformity:', p
    f.write(str(ks) + ', ' + str(p) + '\n')
    test.pvalues.sort()
    y = [(i + 1.) / len(test.pvalues) for i in range(len(test.pvalues))]
    plt.step([0.0] + test.pvalues + [1.0], [0.0] + y + [1.0], color='blue')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('$x$', fontsize=32)
    plt.ylabel('$S_n(x)$', fontsize=32)
    fontprop = {'size': 20}
    a = plt.gca()
    rc('font', **fontprop)
    a.set_xticklabels(a.get_xticks(), fontprop)
    a.set_yticklabels(a.get_yticks(), fontprop)
    plt.tight_layout()
    plt.savefig('RCC_distinct_B.pdf')


    plt.figure()
    test = RCC_tester(N_s=10, N_t=10, C=10)
    ks, p = test.ks_test(n=10000, start_seed=0, control=True)
    print 'p-value of KS-test of uniformity:', p
    f.write(str(ks) + ', ' + str(p) + '\n')
    test.pvalues.sort()
    y = [(i + 1.) / len(test.pvalues) for i in range(len(test.pvalues))]
    plt.step([0.0] + test.pvalues + [1.0], [0.0] + y + [1.0], color='blue')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('$x$', fontsize=32)
    plt.ylabel('$S_n(x)$', fontsize=32)
    fontprop = {'size': 20}
    a = plt.gca()
    rc('font', **fontprop)
    a.set_xticklabels(a.get_xticks(), fontprop)
    a.set_yticklabels(a.get_yticks(), fontprop)
    plt.tight_layout()
    plt.savefig('RCC_distinct_C.pdf')


    plt.figure()
    test = RCC_tester(N_s=10, N_t=100, C=10)
    ks, p = test.ks_test(n=10000, start_seed=0, control=True)
    print 'p-value of KS-test of uniformity:', p
    f.write(str(ks) + ', ' + str(p) + '\n')
    test.pvalues.sort()
    y = [(i + 1.) / len(test.pvalues) for i in range(len(test.pvalues))]
    plt.step([0.0] + test.pvalues + [1.0], [0.0] + y + [1.0], color='blue')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('$x$', fontsize=32)
    plt.ylabel('$S_n(x)$', fontsize=32)
    fontprop = {'size': 20}
    a = plt.gca()
    rc('font', **fontprop)
    a.set_xticklabels(a.get_xticks(), fontprop)
    a.set_yticklabels(a.get_yticks(), fontprop)
    plt.tight_layout()
    plt.savefig('RCC_distinct_D.pdf')




    f.close()
    plt.cla()
    plt.close('all')
