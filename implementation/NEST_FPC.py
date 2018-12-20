'''
@author: Daniel Hjertholm

Tests for network with fixed connection probability for all
possible connections, created by NEST.
'''

import numpy
import numpy.random as rnd
import nest
import nest.topology as topo

from testsuite.FPC_test import FPCTester


class NEST_FPCTester(FPCTester):
    '''
    Tests for network with fixed connection probability for all
    possible connections, created by NEST.
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

        self._L = 1.
        maskdict = {'rectangular': {'lower_left': [-self._L / 2.] * 2,
                                    'upper_right': [self._L / 2.] * 2}}
        kernel = self._p
        self._conndict = {'connection_type': 'divergent',
                          'mask': maskdict, 'kernel': kernel}

    def _reset(self, seed):
        '''
        Reset simulator and seed the PRNGs.

        Parameters
        ----------
            seed: PRNG seed value.
        '''

        nest.ResetKernel()

        if seed is None:
            seed = rnd.randint(10 ** 10)
        seed = 3 * seed # Reduces probability of overlapping seed values.
        rnd.seed(seed)
        nest.SetKernelStatus({'rng_seeds': [seed + 1],
                              'grng_seed': seed + 2})

    def _build(self):
        '''Create populations.'''

        pos = zip((0.,) * self._N_s, (0.,) * self._N_s)
        ldict_s = {'elements': 'iaf_neuron', 'positions': pos,
                   'extent': [self._L] * 2, 'edge_wrap': True}
        pos = zip((0.,) * self._N_t, (0.,) * self._N_t)
        ldict_t = {'elements': 'iaf_neuron', 'positions': pos,
                   'extent': [self._L] * 2, 'edge_wrap': True}
        self._ls = topo.CreateLayer(ldict_s)
        self._lt = topo.CreateLayer(ldict_t)

    def _connect(self):
        '''Connect populations.'''

        topo.ConnectLayers(self._ls, self._lt, self._conndict)

    def _degrees(self):
        '''Return list of degrees.'''

        connections = nest.GetConnections(source=nest.GetLeaves(self._ls)[0])
        i = 0 if self._degree == 'out' else 1
        connections = [conn[i] for conn in connections]
        return self._counter(connections)


class InDegreeTester(NEST_FPCTester):
    '''
    Tests for the in-degree distribution of networks with fixed connection 
    probability for all possible connections, created by NEST.
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
    probability for all possible connections, created by NEST.
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
