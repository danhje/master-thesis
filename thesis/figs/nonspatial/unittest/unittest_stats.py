'''
@author: Daniel Hjertholm

Unittests for RandomConvergentConnect and RandomDivergentConnect.
'''

import unittest

from test_RCC import RCC_tester
from test_RDC import RDC_tester


class RCDCTestCase(unittest.TestCase):
    '''Statistical tests for Random{Con,Di}vergentConnect.'''

    def setUp(self):
        '''Set test parameters and critical values.'''

        # Small network
        self.N_small = 10   # Number of nodes
        self.C_small = 100  # Average number of connections per node
        self.n_small = 1000 # Number of times to repeat test

        # Medium sized network
        self.N_medium = 100 # Number of nodes
        self.C_medium = 100 # Average number of connections per node
        self.n_medium = 100 # Number of times to repeat test

        # Large network
        self.N_large = 1000 # Number of nodes
        self.C_large = 1000 # Average number of connections per node
        self.n_large = 100  # Number of times to repeat test

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

        chi, p = test.chi_squared_test(msd=None)

        if self.alpha1_lower < p < self.alpha1_upper:
            return True
        else:
            ks, p = test.two_level_test(n_runs=n_runs, start_seed=None)
            return True if p > self.alpha2 else False

    def test_RCC_small(self):
        '''Statistical test of RandomConvergentConnect with a small network'''

        test = RCC_tester(N_s=self.N_small, N_t=self.N_small, C=self.C_small)
        passed = self.adaptive_test(test, n_runs=self.n_small)
        self.assertTrue(passed, 'RandomConvergentConnect did not ' \
                        'pass the statistical test procedure.')

    def test_RCC_large(self):
        '''Statistical test of RandomConvergentConnect with a large network'''

        test = RCC_tester(N_s=self.N_large, N_t=self.N_large, C=self.C_large)
        passed = self.adaptive_test(test, n_runs=self.n_large)
        self.assertTrue(passed, 'RandomConvergentConnect did not ' \
                        'pass the statistical test procedure.')

    def test_RCC_threaded(self):
        '''Statistical test of RandomConvergentConnect with 4 threads'''

        test = RCC_tester(N_s=self.N_medium, N_t=self.N_medium, C=self.C_medium,
                          threads=4)
        passed = self.adaptive_test(test, n_runs=self.n_medium)
        self.assertTrue(passed, 'RandomConvergentConnect did not ' \
                        'pass the statistical test procedure.')

    def test_RDC_small(self):
        '''Statistical test of RandomDivergentConnect with a small network'''

        test = RDC_tester(N_s=self.N_small, N_t=self.N_small, C=self.C_small)
        passed = self.adaptive_test(test, n_runs=self.n_small)
        self.assertTrue(passed, 'RandomDivergentConnect did not ' \
                        'pass the statistical test procedure.')

    def test_RDC_large(self):
        '''Statistical test of RandomDivergentConnect with a large network'''

        test = RDC_tester(N_s=self.N_large, N_t=self.N_large, C=self.C_large)
        passed = self.adaptive_test(test, n_runs=self.n_large)
        self.assertTrue(passed, 'RandomDivergentConnect did not ' \
                        'pass the statistical test procedure.')

    def test_RDC_threaded(self):
        '''Statistical test of RandomDivergentConnect with 4 threads'''

        test = RDC_tester(N_s=self.N_medium, N_t=self.N_medium, C=self.C_medium,
                          threads=4)
        passed = self.adaptive_test(test, n_runs=self.n_medium)
        self.assertTrue(passed, 'RandomDivergentConnect did not ' \
                        'pass the statistical test procedure.')


def suite():
    suite = unittest.makeSuite(RCDCTestCase, 'test')
    return suite


if __name__ == '__main__':
  f = 0
  e = 0
  r = 300
  for i in range(r):
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite())
    f += len(res.failures)
    e += len(res.errors)

  fil = open('unittest_stats.txt', 'a')
  fil.write('Runs: %d. Errors: %d. Fails: %d.\n' % (6*r, e, f))
  fil.close()



