'''
@author: Daniel Hjertholm

Tests for network with fixed connection probability for all
possible connections, created by PyNN.
'''

import numpy.random as rnd
import pyNN.nest as sim

from testsuite.FPC_test import FPCTester


class pyNN_FPCTester(FPCTester):
    '''
    Tests for network with fixed connection probability for all
    possible connections, created by PyNN.
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

        sim.nest.set_verbosity('M_FATAL')

        FPCTester.__init__(self, N_s=N_s, N_t=N_t, p=p, e_min=e_min)

    def _reset(self, seed):
        '''
        Reset simulator and seed the PRNGs.

        Parameters
        ----------
            seed: PRNG seed value.
        '''

        sim.end()
        sim.setup()

        # Set PRNG seed values:
        if seed is None:
            seed = rnd.randint(10 ** 10)
        seed = 2 * seed
        rnd.seed(seed)
        self._rng = sim.NumpyRNG(seed=seed + 1)

    def _build(self):
        '''Create populations.'''

        self._source_pop = sim.Population(self._N_s, sim.IF_cond_exp)
        self._target_pop = sim.Population(self._N_t, sim.IF_cond_exp)

    def _connect(self):
        '''Connect populations.'''

        self._proj = sim.Projection(self._source_pop, self._target_pop,
                                    sim.FixedProbabilityConnector(self._p),
                                    rng=self._rng)

    def _degrees(self):
        '''Return list of degrees.'''

        connections = ([c.target for c in self._proj.connections.__iter__()]
                       if self._degree == 'in' else
                       [c.source for c in self._proj.connections.__iter__()])
        return self._counter(connections)


class InDegreeTester(pyNN_FPCTester):
    '''
    Tests for the in-degree distribution of networks with fixed connection 
    probability for all possible connections, created by PyNN.
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
        pyNN_FPCTester.__init__(self, N_s, N_t, p, e_min)


class OutDegreeTester(pyNN_FPCTester):
    '''
    Tests for the out-degree distribution of networks with fixed connection 
    probability for all possible connections, created by PyNN.
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
        pyNN_FPCTester.__init__(self, N_s, N_t, p, e_min)


if __name__ == '__main__':
    test = InDegreeTester(N_s=30, N_t=100, p=0.5)
    ks, p = test.two_level_test(n_runs=100, start_seed=0, control=False)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()
