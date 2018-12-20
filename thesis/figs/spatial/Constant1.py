'''
Created on Mar 9, 2013
@author: danhje

Tests for ConnectLayers. 
'''

import numpy as np
import numpy.random as rnd
import scipy.integrate
import scipy.stats
import matplotlib.pyplot as plt
import nest
import nest.topology as topo


class ConnectLayers_tester(object):
    '''Class used for testing ConnectLayers.'''

    def __init__(self, L, N, function_name, function_params=None,
                 source_pos=(0., 0.), msd=None, threads=1):
        '''
        Initialize an ConnectLayers_tester object.

        Sets up the experiment, and defines network layer and 
        connection parameters .

        Parameters
        ----------
            L              : Side length of square layer.
            N              : Number of  nodes.
            function_name  : Name of distance dependent probability 
                             function (kernel) to test.
            function_params: Parameters for kernel function. Must be
                             such that 0 <= p(D) <= 1 for all
                             distances D in [0, L/sqrt(2)]. If 
                             omitted, sensible defaults are calculated
                             based on layer size.
            source_pos     : Source node position. Default is center.
            msd            : Master PRNG seed. Default is None.
            threads        : Number of local threads. Default is 1.
        '''

        self.threads = threads
        self.msd = msd
        self.L = float(L)
        self.N = N
        self.function_name = function_name

        kernels = {
            'constant': self._constant,
            'linear': self._linear,
            'exponential': self._exponential,
            'gaussian': self._gauss}
        default_params = {
            'constant': 1.0,
            'linear': {'a':-np.sqrt(2) / self.L, 'c': 1.0},
            'exponential': {'a':1.0, 'c': 0.0,
                'tau':-self.L / (np.sqrt(2) * np.log((.1 - 0) / 1))},
            'gaussian': {'p_center': 1., 'sigma': self.L / 4.,
                         'mean': 0., 'c': 0.}}

        self.params = default_params[function_name]
        if function_params is not None:
            if self.function_name == 'constant':
                self.params = function_params
            else:
                self.params.update(function_params)

        self.function = kernels[function_name]

        self.ldict_s = {'elements': 'iaf_neuron',
                        'positions': [source_pos],
                        'extent': [self.L, self.L],
                        'edge_wrap': True}

        if msd is not None:
            rnd.seed(msd)
        x = rnd.uniform(-self.L / 2., self.L / 2., self.N)
        y = rnd.uniform(-self.L / 2., self.L / 2., self.N)
        pos = zip(x, y)
        self.ldict_t = {'elements': 'iaf_neuron',
                        'positions': pos,
                        'extent': [self.L, self.L],
                        'edge_wrap': True}
        self.mask = {'rectangular': {'lower_left': [-self.L / 2.,
                                                    - self.L / 2.],
                                     'upper_right': [self.L / 2.,
                                                     self.L / 2.]}}
        if function_name == 'constant':
            self.kernel = self.params
        else:
            self.kernel = {self.function_name: self.params}
        self.conndict = {'connection_type': 'divergent',
                         'mask': self.mask,
                         'kernel': self.kernel}

    def _reset(self):
        '''Reset the NEST kernel and set PRNG seed values.'''

        nest.ResetKernel()
        nest.SetKernelStatus({"local_num_threads": self.threads})
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')

        # Set PRNG seed values:
        if self.msd is None:
            msd = rnd.randint(1000000)
        else:
            msd = self.msd + 1 # The first was used in __init__.
        msdrange = range(msd, msd + self.n_vp)
        nest.SetKernelStatus({'grng_seed': msd + self.n_vp,
                              'rng_seeds': msdrange})

    def _connect_layers(self):
        '''Create and connect layers.'''

        self.ls = topo.CreateLayer(self.ldict_s)
        self.lt = topo.CreateLayer(self.ldict_t)

        self.driver = topo.FindCenterElement(self.ls)
        topo.ConnectLayers(self.ls, self.lt, self.conndict)

    def _constant(self, D):
        '''Constant kernel function.'''

        return self.params

    def _linear(self, D):
        '''Linear kernel function.'''

        return self.params['c'] + self.params['a'] * D

    def _gauss(self, D):
        '''Gaussian kernel function.'''

        return (self.params['c'] + self.params['p_center'] *
                np.e ** -((D - self.params['mean']) ** 2 /
                          (2. * self.params['sigma'] ** 2)))

    def _exponential(self, D):
        '''Exponential kernel function.'''

        return (self.params['c'] + self.params['a'] *
                np.e ** (-D / self.params['tau']))

    def _pdf(self, D):
        '''Probability density function (PDF). 
        
        Parameters
        ----------
            D: Distance in interval [0, L/sqrt(2)].
            
        Return values
        -------------
            PDF(D)
        '''

        if D <= self.L / 2.:
            return max(0., min(1., self.function(D))) * np.pi * D
        elif self.L / 2. < D <= self.L / np.sqrt(2):
            return max(0., (min(1., self.function(D))) * D *
                    (np.pi - 4. * np.arccos(self.L / (D * 2.))))
        else:
            return 0.

    def _cdf(self, D):
        '''Cumulative distribution function (CDF). 
        
        Parameters
        ----------
            D: Iterable of distances in interval [0, L/sqrt(2)].
            
        Return values
        -------------
            List of CDF(d) for d in D.
        '''

        cdf = []
        last_d = 0.
        for d in D:
            cdf.append(scipy.integrate.quad(self._pdf, last_d, d)[0])
            last_d = d

        top = scipy.integrate.quad(self._pdf, 0, self.L / np.sqrt(2))[0]
        # Stored in case the CDF is to be plotted later.
        self.cdf_list = np.cumsum(cdf) / top
        return self.cdf_list

    def ks_test(self, show_network=False, show_CDF=False,
                show_PDF=False, histogram_bins=100):
        '''
        Perform a Kolmogorov-Smirnov GOF test on the distribution
        of distances to connected nodes. 
        
        Parameters
        ----------
            show_network  : Specify whether network plot should
                            be displayed.
            show_CDF      : Specify whether CDF should be displayed.
            show_PDF      : Specify whether PDF should be displayed.
            histogram_bins: Number of histogram bins for PDF plot.
        
        Return values
        -------------
            KS statistic.
            p-value from KS test.
        '''

        self._reset()
        self._connect_layers()

        connections = nest.GetConnections(source=self.driver)
        target_nodes = [conn[1] for conn in connections]
        if len(target_nodes) == 0: return None, None
        dist = topo.Distance(self.driver, target_nodes)
        dist.sort()

        ks, p = scipy.stats.kstest(dist, self._cdf,
                                   alternative='two_sided')

        if show_network:
            # Adjust size of nodes in plot based on number of nodes.
            nodesize = 1 #max(1, int(round(111. / 11 - self.N / 11000.)))

            fig = plt.figure(figsize=(6,6), frameon=False)
            ax = plt.gca()
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            topo.PlotLayer(self.lt, fig=fig, nodesize=nodesize,
                                 nodecolor='grey')

            # Only the gaussian kernel can be plotted.
            if self.function_name == 'gaussian':
                topo.PlotTargets(self.driver, self.lt, fig=fig,
                                 mask=self.mask, kernel=self.kernel,
                                 mask_color='purple',
                                 kernel_color='blue',
                                 src_size=50, src_color='black',
                                 tgt_size=nodesize, tgt_color='red')
            else:
                topo.PlotTargets(self.driver, self.lt, fig=fig,
                             src_size=50, src_color='black',
                             tgt_size=nodesize, tgt_color='red')
            plt.tight_layout()
            plt.savefig(filename_base + '_network.png', bbox_inches='tight', pad_inches=0)            
            #plt.savefig(filename_base + '_network.pdf', bbox_inches='tight', pad_inches=0)

        if show_CDF:
            plt.figure()
            plt.plot(dist, self.cdf_list, '-', color='black', linewidth=2, zorder=1)
            plt.xlabel('$D$', fontsize=28)
            plt.ylabel('$F(D)$', fontsize=28)
        
            fontprop = {'size': 20}
            a = plt.gca()
            from matplotlib import rc
            rc('font', **fontprop)
            a.set_xticklabels(a.get_xticks(), fontprop)
            a.set_yticklabels(a.get_yticks(), fontprop)
            plt.tight_layout()
            plt.savefig(filename_base + '_CDF.pdf', bbox_inches='tight')

        if show_PDF:
            plt.figure()
            x = np.linspace(0, self.L / np.sqrt(2), 1000)
            area = scipy.integrate.quad(self._pdf, 0,
                                        self.L / np.sqrt(2))[0]
            y = np.array([self._pdf(D) for D in x]) / area
            plt.plot(x, y, color='black', linewidth=2, zorder=1)
            plt.xlabel('$D$', fontsize=28)
            plt.ylabel('$f(D)$', fontsize=28)

            fontprop = {'size': 20}
            a = plt.gca()
            from matplotlib import rc
            rc('font', **fontprop)
            a.set_xticklabels(a.get_xticks(), fontprop)
            a.set_yticklabels(a.get_yticks(), fontprop)
            plt.tight_layout()
            plt.savefig(filename_base + '_PDF.pdf', bbox_inches='tight')

        return ks, p


if __name__ == "__main__":
    filename_base = __file__.split('.')[0]
    test = ConnectLayers_tester(L=1.0, N=100000, msd=0,
                                function_name='constant',
                                function_params=0.5)
    ks, p = test.ks_test(show_network=True,
                         show_PDF=True,
                         show_CDF=True)

