'''
@author: Daniel Hjertholm

Tests for fan-in / -out networks created by PyNN. 
'''

import numpy.random as rnd
import pyNN.nest as sim

from testsuite.fan_test import FanTester


class pyNN_FanTester(FanTester):
    '''Tests for fan-in / -out networks created by PyNN.'''

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

        sim.nest.set_verbosity('M_FATAL')

        FanTester.__init__(self, N_s=N_s, N_t=N_t, C=C, e_min=e_min)

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

        if self._fan == 'in':
            self._p = sim.Projection(self._source_pop, self._target_pop,
                                     sim.FixedNumberPreConnector(n=self._C),
                                     rng=self._rng)
        else:
            self._p = sim.Projection(self._source_pop, self._target_pop,
                                     sim.FixedNumberPostConnector(n=self._C),
                                     rng=self._rng)

    def _degrees(self):
        '''Return list of degrees.'''

        connections = ([c.source for c in self._p.connections.__iter__()]
                       if self._fan == 'in' else
                       [c.target for c in self._p.connections.__iter__()])
        return self._counter(connections)


class FanInTester(pyNN_FanTester):
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
        pyNN_FanTester.__init__(self, N_s, N_t, C, e_min=e_min)


class FanOutTester(pyNN_FanTester):
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
        pyNN_FanTester.__init__(self, N_s, N_t, C, e_min=e_min)


if __name__ == '__main__':
    test = FanInTester(N_s=100, N_t=100, C=10)
    ks, p = test.two_level_test(n_runs=100, start_seed=0)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()
