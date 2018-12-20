'''
@author: Daniel Hjertholm

Tests for networks with fixed connection probability for all
possible connections.
'''

import numpy
import numpy.random as rnd
import scipy.stats
import matplotlib.pyplot as plt


class FPCTester(object):
    '''
    Tests for networks with fixed connection probability for all
    possible connections.
    '''

    def __init__(self, N_s, N_t, p, e_min=5):
        '''
        Construct a test object.

        Parameters
        ----------
            N_s  : Number of nodes in source population.
            N_t  : Number of nodes in target population.
            p    : Connection probability.
            e_min: Minimum expected number of observations in each bin.
        '''

        self._N_s = N_s
        self._N_t = N_t
        self._p = p
        self._e_min = e_min
        self._data = self._expected()

    def _expected(self):
        '''
        Calculate expected degree distribution. 
        
        Degrees with expected number of observations blow e_min are combined
        into larger bins.
        
        Return values
        -------------
            2D array. The four columns contain degree, 
            expected number of observation, actual number observations, and
            the number of bins combined.   
        '''

        n = self._N_s if self._degree == 'in' else self._N_t
        n_p = self._N_t if self._degree == 'in' else self._N_s
        p = self._p

        mid = int(round(n * p))

        # Combine from front.
        data_front = []
        cumexp = 0.0
        bins_combined = 0
        for degree in range(mid):
            cumexp += scipy.stats.binom.pmf(degree, n, p) * n_p
            bins_combined += 1
            if cumexp < self._e_min:
                if degree == mid - 1:
                    if len(data_front) == 0:
                        raise RuntimeWarning('Not enough data')
                    deg, exp, obs, num = data_front[-1]
                    data_front[-1] = (deg, exp + cumexp, obs,
                                      num + bins_combined)
                else:
                    continue
            else:
                data_front.append((degree - bins_combined + 1, cumexp, 0,
                                   bins_combined))
                cumexp = 0.0
                bins_combined = 0

        # Combine from back.
        data_back = []
        cumexp = 0.0
        bins_combined = 0
        for degree in reversed(range(mid, n + 1)):
            cumexp += scipy.stats.binom.pmf(degree, n, p) * n_p
            bins_combined += 1
            if cumexp < self._e_min:
                if degree == mid:
                    if len(data_back) == 0:
                        raise RuntimeWarning('Not enough data')
                    deg, exp, obs, num = data_back[-1]
                    data_back[-1] = (degree, exp + cumexp, obs,
                                      num + bins_combined)
                else:
                    continue
            else:
                data_back.append((degree, cumexp, 0, bins_combined))
                cumexp = 0.0
                bins_combined = 0
        data_back.reverse()

        return numpy.array(data_front + data_back)

    def _reset(self):
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
        counts = [0] * self._N_t if self._degree == 'in' else [0] * self._N_s
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

    def _generate_binomial_degrees(self, seed):
        '''
        Instead of using an actual simulator connection algorithm, this method
        returns data with the expected binomial distribution.

        Parameters
        ----------
            seed: PRNG seed value.

        Return values
        -------------
            list containing data drawn from a biinomial distribution.
        '''

        self._reset(seed)

        dist = (rnd.binomial(self._N_s, self._p, self._N_t)
                if self._degree == 'in' else
                rnd.binomial(self._N_t, self._p, self._N_s))
        return dist

    def chi_squared_test(self, seed=None, control=False):
        '''
        Create a single network and compare the resulting degree distribution
        with the expected distribution using Pearson's chi-squared GOF test.

        Parameters
        ----------
            seed   : PRNG seed value.
            control: Boolean value. If True, _generate_binomial_degrees will
                     be used instead of _get_degrees.

        Return values
        -------------
            chi-squared statistic.
            p-value from chi-squared test.
        '''

        if control:
            degrees = self._generate_binomial_degrees(seed)
        else:
            degrees = self._get_degrees(seed)

        observed = {}
        for degree in degrees:
            if not degree in observed:
                observed[degree] = 1
            else:
                observed[degree] += 1

        # Add observations to data structure, combining multiple observations
        # where necessary.
        self._data[:, 2] = 0.0
        for row in self._data:
            for i in range(int(row[3])):
                deg = int(row[0]) + i
                if deg in observed:
                    row[2] += observed[deg]

        assert (sum(self._data[:, 3]) == self._N_t + 1 if self._degree == 'out'
                else sum(self._data[:, 3]) == self._N_s + 1), 'Something is wrong'


        # ddof: adjustment to the degrees of freedom. df = k-1-ddof
        return scipy.stats.chisquare(numpy.array(self._data[:, 2]),
                                     numpy.array(self._data[:, 1]), ddof=0)

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
            control   : Boolean value. If True, _generate_binomial_degrees
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
