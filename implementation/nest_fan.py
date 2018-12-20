'''
@author: Daniel Hjertholm

Tests for fan-in / -out networks created by NEST. 
'''

import numpy.random as rnd
import nest

from testsuite.fan_test import FanTester


class NEST_FanTester(FanTester):
    '''Tests for fan-in / -out networks created by NEST.'''

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

        nest.set_verbosity('M_FATAL')

        FanTester.__init__(self, N_s=N_s, N_t=N_t, C=C, e_min=e_min)

    def _reset(self, seed):
        '''
        Reset the simulator and seed the PRNGs.

        Parameters
        ----------
            seed: PRNG seed value.
        '''

        nest.ResetKernel()

        # Set PRNG seed values:
        if seed == None:
            seed = rnd.randint(10 ** 10)
        seed = 3 * seed
        rnd.seed(seed)
        nest.SetKernelStatus({'grng_seed': seed + 1,
                              'rng_seeds': [seed + 2]})

    def _build(self):
        '''Create populations.'''

        self._source_pop = nest.Create('iaf_neuron', self._N_s)
        self._target_pop = nest.Create('iaf_neuron', self._N_t)

    def _connect(self):
        '''Connect populations.'''

        if self._fan == 'in':
            nest.RandomConvergentConnect(self._source_pop, self._target_pop,
                self._C, options={'allow_multapses': True})
        elif self._fan == 'out':
            nest.RandomDivergentConnect(self._source_pop, self._target_pop,
                self._C, options={'allow_multapses': True})

    def _degrees(self):
        '''Return list of degrees.'''

        connections = nest.GetConnections(source=self._source_pop)
        i = 0 if self._fan == 'in' else 1
        connections = [conn[i] for conn in connections]
        return self._counter(connections)


class FanInTester(NEST_FanTester):
    '''Tests for fan-in networks created by NEST.'''

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
        NEST_FanTester.__init__(self, N_s, N_t, C, e_min)


class FanOutTester(NEST_FanTester):
    '''Tests for fan-out networks created by NEST.'''

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
        NEST_FanTester.__init__(self, N_s, N_t, C, e_min)


if __name__ == '__main__':
    test = FanInTester(N_s=100, N_t=100, C=100)
    ks, p = test.two_level_test(n_runs=100, start_seed=0)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()
