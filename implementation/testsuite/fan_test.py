'''
@author: Daniel Hjertholm

Tests for fan-in / -out networks.
'''

import numpy
import numpy.random as rnd
import scipy.stats
import matplotlib.pyplot as plt


class FanTester(object):
    '''Tests for fan-in / -out networks.'''

    def __init__(self, N_s, N_t, C, e_min=10):
        '''
        Construct a test object.

        Parameters
        ----------
            N_s  : Number of nodes in source population.
            N_t  : Number of nodes in target population.
            C    : In-degree (number of connections per target neuron).
            e_min: Minimum expected number of observations in each bin.
        '''

        self._N_s = N_s
        self._N_t = N_t
        self._N_d = N_t if self._fan == 'in' else N_s # of driver nodes.
        self._N_p = N_s if self._fan == 'in' else N_t # of pool nodes.
        self._C = C
        self._e_min = e_min

        expected_degree = self._N_d * self._C / float(self._N_p)
        if expected_degree < self._e_min:
            raise RuntimeWarning(
                'Expected degree (%.2f) is less than e_min (%.2f). ' \
                'Results may be unreliable' % \
                (expected_degree, self._e_min))

        self._expected = [expected_degree] * self._N_p

    def _reset(self, seed):
        '''Reset simulator and seed PRNGs.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _build(self):
        '''Create populations.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _connect(self):
        '''Connect populations.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _degrees(self):
        '''Return list of degrees.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

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

        start = min(x)
        counts = [0] * self._N_p
        for elem in x:
            counts[elem - start] += 1

        return counts

    def _get_degrees(self, seed):
        '''
        Reset the simulator, create populations, connect them, and retrieve the
        resulting degrees.

        Parameters
        ----------
            seed: PRNG seed value.

        Return values
        -------------
            list containing the degrees of the nodes in the pool.
        '''

        self._reset(seed)
        self._build()
        self._connect()

        return self._degrees()

    def _generate_multinomial_degrees(self, seed):
        '''
        Instead of using an actual simulator connection algorithm, this method
        returns data with the expected multinomial distribution.

        Parameters
        ----------
            seed: PRNG seed value.

        Return values
        -------------
            list containing data drawn from a multinomial distribution.
        '''

        self._reset(seed)

        con = rnd.randint(0, self._N_p, self._N_d * self._C)
        degrees = self._counter(con)

        return degrees

    def chi_squared_test(self, seed=None, control=False):
        '''
        Create a single network and compare the resulting degree distribution
        with the expected distribution using Pearson's chi-squared GOF test.

        Parameters
        ----------
            seed   : PRNG seed value.
            control: Boolean value. If True, _generate_multinomial_degrees will
                     be used instead of _get_degrees.

        Return values
        -------------
            chi-squared statistic.
            p-value from chi-squared test.
        '''

        if control:
            degrees = self._generate_multinomial_degrees(seed)
        else:
            degrees = self._get_degrees(seed)

        # ddof: adjustment to the degrees of freedom. df = k-1-ddof
        return scipy.stats.chisquare(numpy.array(degrees),
                                     numpy.array(self._expected), ddof=0)

    def two_level_test(self, n_runs, start_seed=None, control=False,
                       verbose=True):
        '''
        Create a network and run chi-squared GOF test n_runs times.
        Test whether resulting p-values are uniformly distributed
        on [0, 1] using the Kolmogorov-Smirnov GOF test.

        Parameters
        ----------
            n_runs    : Number of times to repeat chi-squared test.
            start_seed: First PRNG seed value.
            control   : Boolean value. If True, _generate_multinomial_degrees
                        will be used instead of _get_degrees.
            verbose   : Boolean value, determining whether to print progress.

        Return values
        -------------
            KS statistic.
            p-value from KS test.
        '''

        self._pvalues = []

        if start_seed == None:
            for i in range(n_runs):
                if verbose: print 'Running test %d of %d.' % (i + 1, n_runs)
                chi, p = self.chi_squared_test(None, control)
                self._pvalues.append(p)
        else:
            end_seed = start_seed + n_runs
            for seed in range(start_seed, end_seed):
                if verbose: print 'Running test %d of %d.' % \
                    (1 + (seed - start_seed), n_runs)
                chi, p = self.chi_squared_test(seed, control)
                self._pvalues.append(p)

        ks, p = scipy.stats.kstest(self._pvalues, 'uniform',
                                   alternative='two_sided')

        return ks, p

    def show_CDF(self):
        '''Plot the cumulative distribution function (CDF) of p-values.'''

        plt.figure()
        ps = sorted(self._pvalues)
        y = [i / float(len(ps))
             for i in range(len(ps))]
        plt.step([0.0] + ps + [1.0], [0.0] + y + [1.0])
        plt.xlabel('P-values')
        plt.ylabel('Empirical distribution function')
        plt.show(block=True)

    def show_histogram(self, bins=100):
        '''
        Draw a histogram of p-values.
        
        Parameters 
        ----------
            bins: Number of histogram bins.
        '''

        plt.figure()
        plt.hist(self._pvalues, bins=bins)
        plt.xlabel('P-values')
        plt.ylabel('Frequency')
        plt.show(block=True)
