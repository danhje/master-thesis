'''
@author: Daniel Hjertholm

Tests for network with fixed connection probability for all
possible connections, created by BRIAN.
'''

import numpy.random as rnd
import random
import brian

from testsuite.FPC_test import FPCTester


class BRIAN_FPCTester(FPCTester):
    '''
    Tests for network with fixed connection probability for all
    possible connections, created by BRIAN.
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

        FPCTester.__init__(self, N_s=N_s, N_t=N_t, p=p, e_min=e_min)

    def _reset(self, seed):
        '''
        Reset simulator and seed the PRNGs.

        Parameters
        ----------
            seed: PRNG seed value.
        '''

        # Set PRNG seed values:
        if seed is None:
            seed = rnd.randint(10 ** 10)
        seed = 3 * seed # Reduces probability of overlapping seed values.
        brian.seed(seed)
        rnd.seed(seed + 1)
        random.seed(seed + 2)

    def _build(self):
        '''Create populations.'''

        eqs = brian.Equations('''
            dV/dt  = ge/ms : volt
            dge/dt = ge/ms : volt
            dgi/dt = ge/ms : volt
            ''')

        pop = brian.NeuronGroup(self._N_s + self._N_t, model=eqs,
                                threshold=brian.mV, reset=brian.mV)
        self._source_pop = pop.subgroup(self._N_s)
        self._target_pop = pop.subgroup(self._N_t)

    def _connect(self):
        '''Connect populations.'''

        self._connection = brian.Connection(self._source_pop, self._target_pop,
                                            'ge', sparseness=self._p,
                                            weight=2 * brian.mV)
        self._connection.compress()

    def _degrees(self):
        '''Return list of degrees.'''

        if self._degree == 'out':
            return [len(s) for s in self._connection]
        else:
            c = self._connection
            targets = []
            for i in range(self._N_s):
                targets.extend([j for j in range(self._N_t) if c[i, j] != 0])
            return self._counter(targets)


class InDegreeTester(BRIAN_FPCTester):
    '''
    Tests for the in-degree distribution of networks with fixed connection 
    probability for all possible connections, created by BRIAN.
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
        BRIAN_FPCTester.__init__(self, N_s, N_t, p, e_min)


class OutDegreeTester(BRIAN_FPCTester):
    '''
    Tests for the out-degree distribution of networks with fixed connection 
    probability for all possible connections, created by BRIAN.
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
        BRIAN_FPCTester.__init__(self, N_s, N_t, p, e_min)


if __name__ == '__main__':
    test = InDegreeTester(N_s=30, N_t=100, p=0.5)
    ks, p = test.two_level_test(n_runs=100, start_seed=0, control=False)
    print 'p-value of KS-test of uniformity:', p
    test.show_CDF()
    test.show_histogram()
