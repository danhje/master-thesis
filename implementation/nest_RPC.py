'''
@author: Daniel Hjertholm

Tests for networks created by NEST, where both sources and targets are
drawn randomly. 
'''

import numpy.random as rnd
import nest

from testsuite.RPC_test import RPCTester


class NEST_RPCTester(RPCTester):
    '''
    Tests for networks created by NEST, where both sources and targets are
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

        nest.set_verbosity('M_FATAL')

        RPCTester.__init__(self, N_s=N_s, N_t=N_t, N=N, e_min=e_min)

        self._param_dict = {'weight_m': 1.0,
                            'weight_s': 0.1,
                            'delay_m' : 1.0,
                            'delay_s' : 0.2}

    def _RandomPopulationConnect(self, pre, post, n, param_dict,
                                synapse_model='static_synapse'):
        '''...'''

        nest.sli_func('RandomPopulationConnectD', pre, post, n, param_dict,
                      '/' + synapse_model, litconv=True)

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

        self._RandomPopulationConnect(self._source_pop, self._target_pop,
                                      self._N, self._param_dict)

    def _degrees(self):
        '''Return list of degrees.'''

        connections = nest.GetConnections(source=self._source_pop)
        i = 0 if self._degree == 'out' else 1
        connections = [conn[i] for conn in connections]
        return self._counter(connections)


class InDegreeTester(NEST_RPCTester):
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
        NEST_RPCTester.__init__(self, N_s, N_t, N, e_min)


class OutDegreeTester(NEST_RPCTester):
    '''
    Tests for the out-degree distribution of networks created by NEST, 
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
        NEST_RPCTester.__init__(self, N_s, N_t, N, e_min)


if __name__ == '__main__':
    test = InDegreeTester(N_s=100, N_t=100, N=10000)
    ks, p = test.two_level_test(n_runs=100, start_seed=0)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()
