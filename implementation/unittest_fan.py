'''
@author: Daniel Hjertholm

Unittests for RandomConvergentConnect and RandomDivergentConnect.
'''

import unittest

from nest_fan import FanInTester, FanOutTester


class RCDCTestCase(unittest.TestCase):
    '''Statistical tests for Random{Con,Di}vergentConnect.'''

    def setUp(self):
        '''Set test parameters and critical values.'''

        self.small = {'N': 10, 'C': 100, 'n': 1000}
        self.medium = {'N': 100, 'C': 100, 'n': 100}
        self.large = {'N': 1000, 'C': 1000, 'n': 100}

        # Critical values
        self.alpha1_lower = 0.025
        self.alpha1_upper = 0.975
        self.alpha2 = 0.05

    def adaptive_test(self, test, n_runs):
        '''
        Create a single network using Random{Con/Di}vergentConnect 
        and run a chi-squared GOF test on the connection distribution.
        If the result is extreme (high or low), run a two-level test.
        
        Parameters
        ----------
            test  : Instance of RCC_tester or RDC_tester class.
            n_runs: If chi-square test fails, test is repeated n_runs times, 
                    and the KS test is used to analyze results. 
        
        Return values
        -------------
            boolean value. True if test was passed, False otherwise.
        '''

        chi, p = test.chi_squared_test(seed=None)

        if self.alpha1_lower < p < self.alpha1_upper:
            return True
        else:
            ks, p = test.two_level_test(n_runs=n_runs, start_seed=None)
            return True if p > self.alpha2 else False

    def run_test(self, N_s, N_t, C, n_runs, conntype):
        '''
        Instantiate test class and run adaptive test.
        
        Parameters
        ----------
            N_s     : Number of source neurons.
            N_t     : Number of target neurons.
            C       : Fixed in- or out-degree.
            n_runs  : Number of re-runs of initial test is not passed.
            conntype: Connection type ('convergent' or 'divergent').
        
        Return values
        -------------
            boolean value. True if test was passed, False otherwise.
        '''

        if conntype == 'convergent':
            test = FanInTester(N_s=N_s, N_t=N_t, C=C)
        elif conntype == 'divergent':
            test = FanOutTester(N_s=N_s, N_t=N_t, C=C)
        else:
            raise AttributeError('conntype must be "convergent" or "divergent"')
        return self.adaptive_test(test, n_runs=n_runs)

    def test_RCC_small(self):
        '''Statistical test of RandomConvergentConnect with a small network'''

        self.assertTrue(self.run_test(self.small['N'], self.small['N'],
                                      self.small['C'], self.small['n'],
                                      'convergent'),
                        'RandomConvergentConnect did not pass the test.')

    def test_RCC_large(self):
        '''Statistical test of RandomConvergentConnect with a large network'''

        self.assertTrue(self.run_test(self.large['N'], self.large['N'],
                                      self.large['C'], self.large['n'],
                                      'convergent'),
                        'RandomConvergentConnect did not pass the test.')

    def test_RDC_small(self):
        '''Statistical test of RandomDivergentConnect with a small network'''

        self.assertTrue(self.run_test(self.small['N'], self.small['N'],
                                      self.small['C'], self.small['n'],
                                      'divergent'),
                        'RandomDivergentConnect did not pass the test.')

    def test_RDC_large(self):
        '''Statistical test of RandomDivergentConnect with a large network'''

        self.assertTrue(self.run_test(self.large['N'], self.large['N'],
                                      self.large['C'], self.large['n'],
                                      'divergent'),
                        'RandomDivergentConnect did not pass the test.')


def suite():
    suite = unittest.makeSuite(RCDCTestCase, 'test')
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

