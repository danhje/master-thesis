'''
@author: Daniel Hjertholm

Tests for networks created by CSA, where both sources and targets are
drawn randomly. 
'''

import numpy.random as rnd
import random
import csa

from testsuite.RPC_test import RPCTester


class CSA_RPCTester(RPCTester):
    '''
    Tests for networks created by CSA, where both sources and targets are
    drawn randomly.
    '''

    def __init__(self, N_s, N_t, N, e_min=10):
        '''
        Construct a test object.

        Parameters
        ----------
            N_s  : Number of nodes in source population.
            N_t  : Number of nodes in target population.
            N    : Total number of connections.
            e_min: Minimum expected number of observations in each bin.
        '''

        RPCTester.__init__(self, N_s=N_s, N_t=N_t, N=N, e_min=e_min)

    def _reset(self, seed):
        '''
        Reset the simulator and seed the PRNGs.

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
        self._cs = csa.cset(csa.random(N=self._N) * finite_set)

    def _degrees(self):
        '''
        Return list of degrees.
        
        Parameters
        ----------
            degree: "in" or "out".
        '''

        i = 0 if self._degree == 'out' else 1
        connections = [c[i] for c in self._cs]
        return self._counter(connections)


class InDegreeTester(CSA_RPCTester):
    '''
    Tests for the in-degree distribution of networks created by NEST, 
    where both sources and targets are drawn randomly.
    '''

    def __init__(self, N_s, N_t, N, e_min=10):
        '''
        Construct a test object.

        Parameters
        ----------
            N_s  : Number of nodes in source population.
            N_t  : Number of nodes in target population.
            N    : Total number of connections.
            e_min: Minimum expected number of observations in each bin.
        '''

        self._degree = 'in'
        CSA_RPCTester.__init__(self, N_s, N_t, N, e_min)


class OutDegreeTester(CSA_RPCTester):
    '''
    Tests for the out-degree distribution of networks created by CSA, 
    where both sources and targets are drawn randomly.
    '''

    def __init__(self, N_s, N_t, N, e_min=10):
        '''
        Construct a test object.

        Parameters
        ----------
            N_s  : Number of nodes in source population.
            N_t  : Number of nodes in target population.
            N    : Total number of connections.
            e_min: Minimum expected number of observations in each bin.
        '''

        self._degree = 'out'
        CSA_RPCTester.__init__(self, N_s, N_t, N, e_min)


if __name__ == '__main__':
    test = InDegreeTester(N_s=100, N_t=100, N=10000)
    ks, p = test.two_level_test(n_runs=100, start_seed=0)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()

