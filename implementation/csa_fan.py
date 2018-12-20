'''
@author: Daniel Hjertholm

Tests for fan-in / -out networks created by CSA. 
'''

import numpy.random as rnd
import random
import csa

from testsuite.fan_test import FanTester


class CSA_FanTester(FanTester):
    '''Tests for fan-in / -out networks created by CSA.'''

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

        FanTester.__init__(self, N_s=N_s, N_t=N_t, C=C, e_min=e_min)

    def _reset(self, seed):
        '''
        Seed the PRNGs.

        Parameters
        ----------
            seed: PRNG seed value.
        '''

        # Set PRNG seed values:
        if seed == None:
            seed = rnd.randint(10 ** 10)
        seed = 2 * seed # Reduces probability of overlapping seed values.
        random.seed(seed) # CSA uses random.
        rnd.seed(seed + 1) # _get_expected_distribution uses numpy.random.

    def _build(self):
        '''Create populations.'''

        pass

    def _connect(self):
        '''Connect populations.'''

        finite_set = csa.cross(xrange(self._N_s), xrange(self._N_t))
        if self._fan == 'in':
            self._cs = csa.cset(csa.random(fanIn=self._C) * finite_set)
        else:
            self._cs = csa.cset(csa.random(fanOut=self._C) * finite_set)

    def _degrees(self):
        '''Return list of degrees.'''

        i = 0 if self._fan == 'in' else 1
        connections = [c[i] for c in self._cs]
        return self._counter(connections)


class FanInTester(CSA_FanTester):
    '''Tests for fan-in networks created by CSA.'''

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

        self._fan = 'in'
        CSA_FanTester.__init__(self, N_s, N_t, C, e_min=e_min)


class FanOutTester(CSA_FanTester):
    '''Tests for fan-out networks created by CSA.'''

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

        self._fan = 'out'
        CSA_FanTester.__init__(self, N_s, N_t, C, e_min=e_min)


if __name__ == '__main__':
    test = FanInTester(N_s=100, N_t=100, C=100)
    ks, p = test.two_level_test(n_runs=100, start_seed=0)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()
