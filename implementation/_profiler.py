"""
Used for profiling.
"""

import cProfile
import pstats
import os
#import mock


from brian_FPC import *

_GPROF2DOT = '/usr/local/bin/gprof2dot'
_DOT = '/opt/local/bin/dot'

if __name__ == '__main__':

    # turn off all graphics by replacing graphics functions with mock functions

    # plt.plot = mock.Mock(return_value=None)
    # topo.PlotTargets = mock.Mock(return_value=None)
    # topo.PlotLayer = mock.Mock(return_value=None)
    # topo.PlotKernel = mock.Mock(return_value=None)

    # initialize setup
    prof_file = 'brian_FPC.prof'

    # profile with output to file, longer run
    cProfile.run('''
test = InDegreeTester(N_s=30, N_t=100, p=0.5)
ks, p = test.two_level_test(n_runs=100, start_seed=0, control=False)
print 'p-value of KS-test of uniformity:', p
    ''',
    prof_file)

    # read profile data from file, output statistics in different orders
    prof = pstats.Stats(prof_file).strip_dirs()


    # print '=' * 80
    # prof.sort_stats('name').print_stats()
    #    
    # print '=' * 80
    # prof.sort_stats('cum').print_stats()
    #
    # print '=' * 80
    prof.sort_stats('time').print_stats()
    #
    # print '=' * 80
    # prof.sort_stats('calls').print_stats()
    #
    # print '=' * 80
    # prof.print_callers()
    #
    # print '=' * 80
    # prof.print_callees()


    # finally, invoke gprof2dot and dot to create a figure
    os.system('{gprof2dot} -f pstats -o {file}.dot {file}'
              .format(gprof2dot=_GPROF2DOT, file=prof_file))
    os.system('{dot} -Tpdf -o {file}.pdf {file}.dot'
              .format(dot=_DOT, file=prof_file))
    print "Profile graph stored as {file}.pdf".format(file=prof_file)


