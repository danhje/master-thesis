'''
@author: Daniel Hjertholm

Tests for network with fixed connection probability for all
possible connections, created by the CSA implementation in NEST.
'''

import numpy.random as rnd
import random
import nest
import csa

from testsuite.FPC_test import FPCTester


class NEST_FPCTester(FPCTester):
    '''
    Tests for network with fixed connection probability for all
    possible connections, created by the CSA implementation in NEST.
    '''

    def __init__(self, N_s, N_t, p, e_min=5):
        '''
        Construct a test object.

        Parameters
        ----------
            N_s   : Number of nodes in source population.
            N_t   : Number of nodes in target population.
            p     : Connection probability.
            e_min : Minimum expected number of observations in each bin.
        '''

        nest.set_verbosity('M_FATAL')

        FPCTester.__init__(self, N_s=N_s, N_t=N_t, p=p, e_min=e_min)

    def _reset(self, seed):
        '''
        Reset simulator and seed the PRNGs.

        Parameters
        ----------
            seed: PRNG seed value.
        '''

        nest.ResetKernel()

        # Set PRNG seed values:
        if seed == None:
            seed = rnd.randint(10 ** 10)
        seed = 4 * seed # Reduces probability of overlapping seed values.
        random.seed(seed) # CSA uses random.
        rnd.seed(seed + 1) # _get_expected_distribution uses numpy.random.
        nest.SetKernelStatus({'grng_seed': seed + 2,
                              'rng_seeds': [seed + 3]})

    def _build(self):
        '''Create populations.'''

        self._source_pop = nest.Create('iaf_neuron', self._N_s)
        self._target_pop = nest.Create('iaf_neuron', self._N_t)

    def _connect(self):
        '''Connect populations.'''

        finite_set = csa.cross(xrange(self._N_s), xrange(self._N_t))
        cs = csa.cset(csa.random(p=self._p) * finite_set)
        nest.CGConnect(self._source_pop, self._target_pop, csa.cset(cs))

    def _degrees(self):
        '''Return list of degrees.'''

        connections = nest.GetConnections(source=self._source_pop)
        i = 0 if self._degree == 'out' else 1
        connections = [conn[i] for conn in connections]
        return self._counter(connections)


class InDegreeTester(NEST_FPCTester):
    '''
    Tests for the in-degree distribution of networks with fixed connection 
    probability for all possible connections, created by the CSA implementation
    in NEST.
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

        self._degree = 'in'
        NEST_FPCTester.__init__(self, N_s, N_t, p, e_min)


class OutDegreeTester(NEST_FPCTester):
    '''
    Tests for the out-degree distribution of networks with fixed connection 
    probability for all possible connections, created by the CSA implementation
    in NEST.
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

        self._degree = 'out'
        NEST_FPCTester.__init__(self, N_s, N_t, p, e_min)


if __name__ == '__main__':
    test = InDegreeTester(N_s=30, N_t=100, p=0.5)
    ks, p = test.two_level_test(n_runs=100, start_seed=0, control=False)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()
