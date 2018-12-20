'''
@author: Daniel Hjertholm

Tests for ConnectLayers. 
'''

import numpy as np
import numpy.random as rnd
import scipy.integrate
import scipy.stats
import matplotlib.pyplot as plt
import nest
import nest.topology as topo


class ConnectLayers2D_tester(object):
    '''Class used for testing ConnectLayers.'''

    def __init__(self, L, N, kernel_name, kernel_params=None,
                 source_pos=None, msd=None, threads=1):
        '''
        Initialize an ConnectLayers2D_tester object.

        Sets up the experiment, and defines network layer and 
        connection parameters .

        Parameters
        ----------
            L            : Side length of square layer.
            N            : Number of  nodes.
            kernel_name  : Name of distance dependent probability 
                           function (kernel) to test.
            kernel_params: Parameters for kernel function. If 
                           omitted, sensible defaults are calculated
                           based on layer size.
            source_pos   : Source node position. Default is center.
            msd          : Master PRNG seed. Default is None.
            threads      : Number of local threads. Default is 1.
        '''

        self.threads = threads
        self.msd = msd
        self.L = float(L)
        self.N = N
        self.kernel_name = kernel_name

        kernels = {
            'constant': self._constant,
            'linear': self._linear,
            'exponential': self._exponential,
            'gaussian': self._gauss}
        default_params = {
            'constant': 1.0,
            'linear': {'a':-np.sqrt(2) / self.L, 'c': 1.0},
            'exponential': {'a':1.0, 'c': 0.0,
                            'tau':-self.L /
                                  (np.sqrt(2) * np.log((.1 - 0) / 1))},
            'gaussian': {'p_center': 1., 'sigma': self.L / 4.,
                         'mean': 0., 'c': 0.}}

        self.params = default_params[kernel_name]
        if kernel_params is not None:
            if self.kernel_name == 'constant':
                self.params = kernel_params
            else:
                self.params.update(kernel_params)

        self.kernel_func = kernels[kernel_name]

        if source_pos is None: source_pos = (0., 0.)
        self.ldict_s = {'elements': 'iaf_neuron', 'positions': [source_pos],
                        'extent': [self.L, self.L], 'edge_wrap': True}

        if msd is not None:
            rnd.seed(msd)
        x = rnd.uniform(-self.L / 2., self.L / 2., self.N)
        y = rnd.uniform(-self.L / 2., self.L / 2., self.N)
        pos = zip(x, y)
        self.ldict_t = {'elements': 'iaf_neuron', 'positions': pos,
                        'extent': [self.L, self.L], 'edge_wrap': True}
        self.mask = {'rectangular': {'lower_left': [-self.L / 2.,
                                                    - self.L / 2.],
                                     'upper_right': [self.L / 2.,
                                                     self.L / 2.]}}
        if kernel_name == 'constant':
            self.kernel = self.params
        else:
            self.kernel = {self.kernel_name: self.params}
        self.conndict = {'connection_type': 'divergent',
                         'mask': self.mask, 'kernel': self.kernel}

    def _constant(self, D):
        '''Constant kernel function.'''

        return self.params

    def _linear(self, D):
        '''Linear kernel function.'''

        return self.params['c'] + self.params['a'] * D

    def _exponential(self, D):
        '''Exponential kernel function.'''

        return (self.params['c'] + self.params['a'] *
                np.e ** (-D / self.params['tau']))

    def _gauss(self, D):
        '''Gaussian kernel function.'''

        return (self.params['c'] + self.params['p_center'] *
                np.e ** -((D - self.params['mean']) ** 2 /
                          (2. * self.params['sigma'] ** 2)))

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

        if D <= self.L / 2.:
            return (max(0., min(1., self.kernel_func(D))) * np.pi * D)
        elif self.L / 2. < D <= self.L / np.sqrt(2):
            return (max(0., min(1., self.kernel_func(D))) *
                    D * (np.pi - 4. * np.arccos(self.L / (D * 2.))))
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

        top = scipy.integrate.quad(self._pdf, 0, self.L / np.sqrt(2))[0]
        normed_cdf = cdf / top

        # Stored in case the CDF is to be plotted later.
        self.cdf_list = normed_cdf

        return normed_cdf

    def _reset(self):
        '''Reset the NEST kernel and set PRNG seed values.'''

        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': self.threads})
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')

        # Set PRNG seed values:
        if self.msd is None:
            msd = rnd.randint(1000000)
        else:
            msd = self.msd + 1 # The first was used in __init__.
        msdrange = range(msd, msd + self.n_vp)
        nest.SetKernelStatus({'grng_seed': msd + self.n_vp,
                              'rng_seeds': msdrange})

    def _build(self):
        '''Create layers.'''

        self.ls = topo.CreateLayer(self.ldict_s)
        self.lt = topo.CreateLayer(self.ldict_t)
        self.driver = topo.FindCenterElement(self.ls)

    def _connect(self):
        '''Connect layers.'''

        topo.ConnectLayers(self.ls, self.lt, self.conndict)

    def _get_distances(self):
        '''
        Create and connect layers, and get distances to connected nodes.
        
        Return values
        -------------
            Ordered list of distances to connected nodes.
        '''

        self._reset()
        self._build()
        self._connect()

        connections = nest.GetConnections(source=self.driver)
        target_nodes = [conn[1] for conn in connections]
        if len(target_nodes) == 0: return None, None
        dist = topo.Distance(self.driver, target_nodes)
        dist.sort()

        return dist

    def _get_distances_control(self):
        '''
        Instead of using NEST's actual connection algorithm, this 
        method creates data with the expected distribution.
        
        Return values
        -------------
            Ordered list of distances to connected nodes.
        '''

        self._reset()
        self._build()

        dist = topo.Distance(self.driver, nest.GetLeaves(self.lt)[0])
        dist = [d for d in dist if rnd.uniform() < self.kernel_func(d)]
        dist.sort()

        return dist

    def ks_test(self, control=False, alternative='two_sided',
                show_network=False, show_CDF=False, show_PDF=False,
                histogram_bins=100):
        '''
        Perform a Kolmogorov-Smirnov GOF test on the distribution
        of distances to connected nodes. 
        
        Parameters
        ----------
            control       : Boolean value. If True, _get_distances_control
                            will be used instead of _get_distances.
            alternative   : Alternative hypothesis. 'two_sided' (default), 
                            'less' or 'greater'.
            show_network  : Specify whether network plot should be displayed.
            show_CDF      : Specify whether CDF should be displayed.
            show_PDF      : Specify whether PDF should be displayed.
            histogram_bins: Number of histogram bins for PDF plot.
        
        Return values
        -------------
            KS statistic.
            p-value from KS test.
        '''

        if control:
            dist = self._get_distances_control()
        else:
            dist = self._get_distances()

        ks, p = scipy.stats.kstest(dist, self._cdf,
                                   alternative=alternative)

        if show_network and not control:
            # Adjust size of nodes in plot based on number of nodes.
            nodesize = max(1, int(round(111. / 11 - self.N / 11000.)))

            fig = topo.PlotLayer(self.lt, nodesize=nodesize,
                                 nodecolor='grey')

            # Only the gaussian kernel can be plotted.
            if self.kernel_name == 'gaussian':
                topo.PlotTargets(self.driver, self.lt, fig=fig,
                                 mask=self.mask, kernel=self.kernel,
                                 mask_color='purple',
                                 kernel_color='blue',
                                 src_size=50, src_color='black',
                                 tgt_size=nodesize, tgt_color='red')
            else:
                topo.PlotTargets(self.driver, self.lt, fig=fig,
                                 mask=self.mask, mask_color='purple',
                                 src_size=50, src_color='black',
                                 tgt_size=nodesize, tgt_color='red')

        if show_CDF:
            plt.figure()
            plt.plot(dist, self.cdf_list, '-', color='black', linewidth=3,
                     label='Theory', zorder=1)
            y = [(i + 1.) / len(dist) for i in range(len(dist))]
            plt.step([0.0] + dist, [0.0] + y, color='red',
                     linewidth=1, label='Empirical', zorder=2)
            plt.ylim(0, 1)

            plt.xlabel('Distance')
            plt.ylabel('CDF')
            plt.legend(loc='center right')

        if show_PDF:
            plt.figure()
            x = np.linspace(0, self.L / np.sqrt(2), 1000)
            area = scipy.integrate.quad(self._pdf, 0, self.L / np.sqrt(2))[0]
            y = np.array([self._pdf(D) for D in x]) / area
            plt.plot(x, y, color='black', linewidth=3, label='Theory', zorder=1)
            plt.hist(dist, bins=histogram_bins, histtype='step',
                     linewidth=1, normed=True, color='red',
                     label='Empirical', zorder=2)
            plt.ylim(ymin=0.)

            plt.xlabel('Distance')
            plt.ylabel('PDF')
            plt.legend(loc='center right')

        if show_CDF or show_PDF or show_network:
            plt.show(block=True)

        return ks, p

    def z_test(self, control=False):
        '''
        Perform a Z-test on the total number of connections.
        
        Parameters
        ---------
            control: Boolean value. If True, _get_distances_control will be used
            instead of _get_distances.
            
        Return values
        -------------
            Standard score (z-score).
            Two-sided p-value.
        '''

        if control:
            num = len(self._get_distances_control())
        else:
            num = len(self._get_distances())

        dist = topo.Distance(self.driver, nest.GetLeaves(self.lt)[0])
        ps = ([max(0., min(1., self.kernel_func(D))) for D in dist])
        exp = sum(ps)
        var = sum([p * (1. - p) for p in ps])
        if var == 0: return np.nan, 1.0
        sd = np.sqrt(var)
        z = abs((num - exp) / sd)
        p = 2. * (1. - scipy.stats.norm.cdf(z))

        return z, p


if __name__ == '__main__':
    test = ConnectLayers2D_tester(L=1.0, N=100000,
                                  kernel_name='gaussian', msd=0)
    ks, p = test.ks_test(show_network=True, show_PDF=True, show_CDF=True)
    print 'KS test statistic:', ks
    print 'p-value of KS-test:', p
    z, p = test.z_test()
    print 'Z-score:', z
    print 'p-value of Z-test:', p

