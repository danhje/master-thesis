'''
@author: Daniel Hjertholm

Tests for spatially structured networks.
'''

import numpy as np
import numpy.random as rnd
import scipy.integrate
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class SpatialTester(object):
    '''Tests for spatially structured networks.'''

    def __init__(self, L, N):
        '''
        Construct a test object.

        Parameters
        ----------
            L: Side length of area / volume.
            N: Number of nodes.
        '''

        self._L = float(L)
        self._N = N
        self._max_dist = (self._L / np.sqrt(2) if self._dimensions == 2
                          else self._L * np.sqrt(3) / 2)

    def _reset(self, seed):
        '''Reset simulator and seed PRNGs.'''

        pass

    def _build(self):
        '''Create populations.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _connect(self):
        '''Connect populations.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _kernel(self, D):
        '''Distance dependent probability function (kernel).'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _positions(self):
        '''Return list of position tuples for all nodes.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _target_positions(self):
        '''Return list of position tuples of all connected target nodes.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _distances(self):
        '''Return list with distances to all nodes.'''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _target_distances(self):
        '''
        Return list with distances from source node to all connected
        target nodes.
        '''

        raise NotImplementedError('This method should be implemented by ' \
                                  'simulator-specific subclass')

    def _pdf(self, D):
        '''
        Unnormalized probability density function (PDF). 
        
        Parameters
        ----------
            D: Distance in interval [0, L/sqrt(2)].
            
        Return values
        -------------
            Unnormalized PDF at distance D.
        '''

        if self._dimensions == 2:
            if D <= self._L / 2.:
                return (max(0., min(1., self._kernel(D))) * np.pi * D)
            elif self._L / 2. < D <= self._max_dist:
                return (max(0., min(1., self._kernel(D))) *
                        D * (np.pi - 4. * np.arccos(self._L / (D * 2.))))
            else:
                return 0.
        elif self._dimensions == 3:
            if D <= self._L / 2.:
                return (max(0., min(1., self._kernel(D))) *
                        4. * np.pi * D ** 2.)
            elif self._L / 2. < D <= self._L / np.sqrt(2):
                return (max(0., min(1., self._kernel(D))) *
                        2. * np.pi * D * (3. * self._L - 4. * D))
            elif self._L / np.sqrt(2) < D <= self._max_dist:
                A = 4. * np.pi * D ** 2.
                C = 2. * np.pi * D * (D - self._L / 2.)
                alpha = np.arcsin(1. / np.sqrt(2. - self._L ** 2. /
                                               (2. * D ** 2.)))
                beta = np.pi / 2.
                gamma = np.arcsin(np.sqrt((1. - .5 * (self._L / D) ** 2.) /
                                          (1. - .25 * (self._L / D) ** 2.)))
                T = D ** 2. * (alpha + beta + gamma - np.pi)
                return (max(0., min(1., self._kernel(D))) *
                        (A + 6. * C * (-1. + 4. * gamma / np.pi) - 48. * T))
            else:
                return 0.

    def _cdf(self, D):
        '''
        Normalized cumulative distribution function (CDF). 
        
        Parameters
        ----------
            D: Iterable of distances in interval [0, L/sqrt(2)]. 
            
        Return values
        -------------
            List of CDF(d) for each distance d in D.
        '''

        cdf = []
        last_d = 0.
        for d in D:
            cdf.append(scipy.integrate.quad(self._pdf, last_d, d)[0])
            last_d = d

        cdf = np.cumsum(cdf)

        top = scipy.integrate.quad(self._pdf, 0, self._max_dist)[0]
        normed_cdf = cdf / top

        return normed_cdf

    def _get_distances(self, seed=None):
        '''
        Create and connect populations, and retrieve distances to connected nodes.
        
        Parameters
        ----------
            seed: PRNG seed value.
        
        Return values
        -------------
            Ordered list of distances to connected nodes.
        '''

        self._control = False

        self._reset(seed)
        self._build()
        self._connect()

        dist = self._target_distances()
        dist.sort()

        return dist

    def _get_expected_distribution(self, seed):
        '''
        Instead of using an actual connection algorithm from a simulator, this 
        method creates data with the expected distribution.
        
        Return values
        -------------
            Ordered list of distances to connected nodes.
        '''

        self._control = True

        self._reset(seed)
        self._build()

        dist = self._distances()
        dist = [d for d in dist if rnd.uniform() < self._kernel(d)]
        dist.sort()

        return dist

    def ks_test(self, control=False, seed=None):
        '''
        Perform a Kolmogorov-Smirnov GOF test on the distribution
        of distances to connected nodes. 
        
        Parameters
        ----------
            control: Boolean value. If True, data with a distribution known to
                     fulfill H0 will be used instead of the simulators
                     connection routine.
            seed   : PRNG seed value.
        
        Return values
        -------------
            KS statistic.
            p-value from KS test.
        '''

        if control:
            self._dist = self._get_expected_distribution(seed)
        else:
            self._dist = self._get_distances(seed)

        ks, p = scipy.stats.kstest(self._dist, self._cdf,
                                   alternative='two_sided')

        return ks, p

    def z_test(self, control=False, seed=None):
        '''
        Perform a Z-test on the total number of connections.
        
        Parameters
        ---------
            control: Boolean value. If True, data with a distribution known to
                     fulfill H0 will be used instead of the simulators
                     connection routine.
            seed   : PRNG seed value.
            
        Return values
        -------------
            Standard score (z-score).
            Two-sided p-value.
        '''

        if control:
            self._dist = self._get_expected_distribution(seed)
        else:
            self._dist = self._get_distances(seed)
        num = len(self._dist)

        dist = self._distances()
        ps = ([max(0., min(1., self._kernel(D))) for D in dist])
        exp = sum(ps)
        var = sum([p * (1. - p) for p in ps])
        if var == 0: return np.nan, 1.0
        sd = np.sqrt(var)
        z = abs((num - exp) / sd)
        p = 2. * (1. - scipy.stats.norm.cdf(z))

        return z, p

    def show_network(self):
        '''Plot nodes in the network.'''

        if self._control:
            return

        # Adjust size of nodes in plot based on number of nodes.
        nodesize = max(0.01, round(111. / 11 - self._N / 1100.))

        figsize = (8, 6) if self._dimensions == 3 else (6, 6)
        fig = plt.figure(figsize=figsize)
        positions = self._positions()
        connected = self._target_positions()
        not_connected = set(positions) - set(connected)

        x1 = [pos[0] for pos in not_connected]
        y1 = [pos[1] for pos in not_connected]
        x2 = [pos[0] for pos in connected]
        y2 = [pos[1] for pos in connected]

        if self._dimensions == 2:
            plt.scatter(x1, y1, s=nodesize, marker='.', color='grey')
            plt.scatter(x2, y2, s=nodesize, marker='.', color='red')
        if self._dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            z1 = [pos[2] for pos in not_connected]
            z2 = [pos[2] for pos in connected]
            ax.scatter(x1, y1, z1, s=nodesize, marker='.', color='grey')
            ax.scatter(x2, y2, z2, s=nodesize, marker='.', color='red')

        plt.show(block=True)

    def show_CDF(self):
        '''
        Plot the cumulative distribution function (CDF) of 
        source-target distances.
        '''

        plt.figure()
        x = np.linspace(0, self._max_dist, 1000)
        cdf = self._cdf(x)
        plt.plot(x, cdf, '-', color='black', linewidth=3,
                 label='Theory', zorder=1)
        y = [(i + 1.) / len(self._dist) for i in range(len(self._dist))]
        plt.step([0.0] + self._dist, [0.0] + y, color='red',
                 linewidth=1, label='Empirical', zorder=2)
        plt.ylim(0, 1)

        plt.xlabel('Distance')
        plt.ylabel('CDF')
        plt.legend(loc='center right')
        plt.show(block=True)

    def show_PDF(self, bins=100):
        '''
        Plot the probability density function (PDF) of source-target distances.
        
        Parameters
        ----------
            bins: Number of histogram bins for PDF plot.
        '''

        plt.figure()
        x = np.linspace(0, self._max_dist, 1000)
        area = scipy.integrate.quad(self._pdf, 0, self._max_dist)[0]
        y = np.array([self._pdf(D) for D in x]) / area
        plt.plot(x, y, color='black', linewidth=3, label='Theory', zorder=1)
        plt.hist(self._dist, bins=bins, histtype='step',
                 linewidth=1, normed=True, color='red',
                 label='Empirical', zorder=2)
        plt.ylim(ymin=0.)

        plt.xlabel('Distance')
        plt.ylabel('PDF')
        plt.legend(loc='center right')
        plt.show(block=True)
