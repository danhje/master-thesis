'''
@author: Daniel Hjertholm

Unittests for 2D and 3D spatially structured networks.
'''

import unittest
import scipy.stats

from nest_spatial import Spatial2DTester, Spatial3DTester


class ConnectLayersTestCase(unittest.TestCase):
    '''Statistical tests for ConnectLayers.'''

    def setUp(self):
        '''Set test parameters and critical values.'''

        self.N = 100000   # Number of nodes
        self.L = 1.0      # Layer size. 
        self.n_runs = 100 # Number of times to repeat test

        # Critical values
        self.alpha1 = 0.01
        self.alpha2 = 0.01

    def ks_test(self, tester_class, kernel_name, kernel_params={}):
        '''
        Create a single network using ConnectLayers, and perform a 
        Kolmogorov-Smirnov  (KS) test on the distribution of source-target
        distances. If the result is suspicious, the test is repeated n_runs 
        times, and the resulting p-values are compared with the expected uniform
        distribution using the KS test.
        
        Parameters
        ----------
            tester_class : ConnectLayers2D_tester or ConnectLayers3D_tester.
            kernel_name  : Name of distance dependent probability 
                           function (kernel) to test.
            kernel_params: Parameters for kernel function. Optional.
        
        Return values
        -------------
            boolean value. True if test was passed, False otherwise.
        '''

        test = tester_class(L=self.L, N=self.N, kernel_name=kernel_name,
                            kernel_params=kernel_params)
        ks, p = test.ks_test()

        if p > self.alpha1:
            return True
        else:
            ps = []
            print ''
            for i in range(self.n_runs):
                print 'Running test %d of %d.' % (i + 1, self.n_runs)
                test = tester_class(L=self.L, N=self.N, kernel_name=kernel_name,
                                    kernel_params=kernel_params,
                                    seed=None)
                ps.append(test.ks_test()[1])
            ks, p = scipy.stats.kstest(ps, 'uniform', alternative='two_sided')
            return True if p > self.alpha2 else False

    def z_test(self, tester_class, kernel_name, kernel_params={}):
        '''
        Create a single network using ConnectLayers, and perform a Z-test on the
        total connection count. If the result is suspicious, the test is
        repeated n_runs times, and the resulting p-values are compared with the
        expected uniform distribution using the KS test.
        
        Parameters
        ----------
            tester_class : ConnectLayers2D_tester or ConnectLayers3D_tester.
            kernel_name  : Name of distance dependent probability 
                           function (kernel) to test.
            kernel_params: Parameters for kernel function. Optional.
        
        Return values
        -------------
            boolean value. True if test was passed, False otherwise.
        '''

        test = tester_class(L=self.L, N=self.N, kernel_name=kernel_name,
                            kernel_params=kernel_params)
        z, p = test.z_test()

        if p > self.alpha1:
            return True
        else:
            ps = []
            print ''
            for i in range(self.n_runs):
                print 'Running test %d of %d.' % (i + 1, self.n_runs)
                test = tester_class(L=self.L, N=self.N,
                                    kernel_name=kernel_name,
                                    kernel_params=kernel_params,
                                    seed=None)
                ps.append(test.z_test()[1])
            z, p = scipy.stats.kstest(ps, 'uniform', alternative='two_sided')
            return True if p > self.alpha2 else False

    def test_2D_constant_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial2DTester, 'constant',
                                     kernel_params=0.5),
                        'ConnectLayers failed to pass the KS test.')

    def test_2D_constant_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial3DTester, 'constant',
                               kernel_params=0.5),
                        'ConnectLayers failed to pass the Z-test')

    def test_2D_linear_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial2DTester, 'linear'),
                        'ConnectLayers failed to pass the KS test.')

    def test_2D_linear_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial2DTester, 'linear'),
                        'ConnectLayers failed to pass the Z-test')

    def test_2D_exponential_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial2DTester, 'exponential'),
                        'ConnectLayers failed to pass the KS test.')

    def test_2D_exponential_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial2DTester, 'exponential'),
                        'ConnectLayers failed to pass the Z-test')

    def test_2D_gaussian_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial2DTester, 'gaussian'),
                        'ConnectLayers failed to pass the KS test.')

    def test_2D_gaussian_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial2DTester, 'gaussian'),
                        'ConnectLayers failed to pass the Z-test')

    def test_3D_constant_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial3DTester, 'constant',
                                     kernel_params=0.5),
                        'ConnectLayers failed to pass the KS test.')

    def test_3D_constant_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial3DTester, 'constant',
                                    kernel_params=0.5),
                        'ConnectLayers failed to pass the Z-test')

    def test_3D_linear_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial3DTester, 'linear'),
                        'ConnectLayers failed to pass the KS test.')

    def test_3D_linear_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial3DTester, 'linear'),
                        'ConnectLayers failed to pass the Z-test')

    def test_3D_exponential_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial3DTester, 'exponential'),
                        'ConnectLayers failed to pass the KS test.')

    def test_3D_exponential_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial3DTester, 'exponential'),
                        'ConnectLayers failed to pass the Z-test')

    def test_3D_gaussian_ks(self):
        '''KS test performed on source-target node distances'''

        self.assertTrue(self.ks_test(Spatial3DTester, 'gaussian'),
                        'ConnectLayers failed to pass the KS test.')

    def test_3D_gaussian_z(self):
        '''Z-test performed on total connection count'''

        self.assertTrue(self.z_test(Spatial3DTester, 'gaussian'),
                        'ConnectLayers failed to pass the Z-test')


def suite():
    suite = unittest.makeSuite(ConnectLayersTestCase, 'test')
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

