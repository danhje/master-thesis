import re
import matplotlib.pyplot as plt
import scipy.optimize as o
import numpy as np


def pf(bias):
    'Plot detection rate vs. N.'

    bias = str(bias)
    f = open('bias_' + bias + '_fails_vs_N.txt', 'r')
    lines = f.readlines()
    f.close()

    Ns = []
    fails_ks = []
    fails_z = []
    for line in lines:
        line = line.replace(' ', '').replace('N:', '').replace('\n', '')
        line = re.split('N:|,KSfails:|,Zfails:', line)
        Ns.append(int(line[0]))
        fails_ks.append(int(line[1]))
        fails_z.append(int(line[2]))

    Ns = np.array(Ns); fails_ks = np.array(fails_ks)/100.; fails_z = np.array(fails_z)/100.
    n = 100.0
    se_ks = 1.96*np.sqrt(fails_ks * (1. - fails_ks) / n)
    se_z = 1.96*np.sqrt(fails_z * (1. - fails_z) / n)

    plt.figure(figsize=(6, 4))

    plt.errorbar(Ns, fails_ks, yerr=se_ks, fmt='bo', label='$\mathrm{KS \, test}$')
    plt.errorbar(Ns, fails_z, yerr=se_z, fmt='rv', label='$Z\mathrm{-test}$')

    plt.legend(loc='lower right')
    plt.xlabel('$N$', fontsize=17)
    plt.ylabel('$\mathrm{Detection \, rate}$', fontsize=17)

    plt.xlim(-5000, max(Ns)+5000)
    plt.ylim(-0.05, 1.05)

    fontprop = {'size': 12}
    ax = plt.gca()
    from matplotlib import rc
    rc('font', **fontprop)
    if max(Ns)/1000 % 7 == 0:
        num_ticks = 7
    else:
        num_ticks = 5
    ax.set_xticks(np.linspace(0, max(Ns), num_ticks+1))

    ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontprop)
    ax.set_yticklabels(ax.get_yticks(), fontprop)

    plt.tight_layout()
    plt.savefig('bias_' + bias + '_plot_fails.pdf', bbox_inches='tight')
    plt.cla()
    plt.close('all')


def pm(bias):
    '''Plot mean p-value vs. N.'''

    bias = str(bias)
    f = open('bias_' + bias + '_pvals_vs_N.txt', 'r')
    lines = f.readlines()
    f.close()

    Ns = []
    ks_means = []
    z_means = []
    ks = []
    z = []
    last_N = None
    lines.append('N:       9999999, seed:   0, num:     0, p_ks: 0.0, p_z: 0.0')
    for line in lines:
        line = line.replace(' ', '').replace('N:', '').replace('\n', '')
        line = re.split('N:|,seed:|,num:|,p_ks:|,p_z:', line)
        N = int(line[0])
        if N != last_N and len(ks) != 0:
            Ns.append(last_N)
            ks_means.append(float(sum(ks)) / len(ks))
            z_means.append(float(sum(z)) / len(z))
            ks = []
            z = []
        last_N = N
        ks.append(float(line[3]))
        z.append(float(line[4]))

    plt.figure(figsize=(6, 4))

    plt.scatter(Ns, ks_means, marker='o', s=10, color='blue', label='$\mathrm{KS \, test}$')
    plt.scatter(Ns, z_means, marker='v', s=10, color='red', label='$Z\mathrm{-test}$')

    plt.legend(loc='upper right')
    plt.xlabel('$N$', fontsize=17)
    plt.ylabel('$\mathrm{Mean \,}p\mathrm{-value}$', fontsize=17)

    plt.xlim(-5000, max(Ns)+5000)
    plt.ylim(-0.05, 1.05)

    fontprop = {'size': 12}
    ax = plt.gca()
    from matplotlib import rc
    rc('font', **fontprop)
    if max(Ns)/1000 % 7 == 0:
        num_ticks = 7
    else:
        num_ticks = 5
    ax.set_xticks(np.linspace(0, max(Ns), num_ticks+1))

    ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontprop)
    ax.set_yticklabels(ax.get_yticks(), fontprop)

    plt.tight_layout()
    plt.savefig('bias_' + bias + '_plot_pvals_mean.pdf', bbox_inches='tight')
    plt.cla()
    plt.close('all')


def p(bias):
    '''Plot p-values vs. N.'''

    bias = str(bias)
    f = open('bias_' + bias + '_pvals_vs_N.txt', 'r')
    lines = f.readlines()
    f.close()

    Ns = []
    ks = []
    z = []
    for i in range(len(lines)):
        line = lines[i].replace(' ', '').replace('N:', '').replace('\n', '')
        line = re.split('N:|,seed:|,num:|,p_ks:|,p_z:', line)
        N = int(line[0])
        Ns.append(N)
        ks.append(float(line[3]))
        z.append(float(line[4]))

    plt.figure(figsize=(6, 4))

    plt.scatter(Ns, ks, marker='o', s=5, color='blue', label='$\mathrm{KS \, test}$')
    plt.scatter(Ns, z, marker='o', s=5, color='red', label='$Z\mathrm{-test}$')

    plt.legend(loc='upper right')
    plt.xlabel('$N$', fontsize=17)
    plt.ylabel('$p\mathrm{-values}$', fontsize=17)

    plt.xlim(-5000, max(Ns)+5000)
    plt.ylim(-0.05, 1.05)

    fontprop = {'size': 12}
    ax = plt.gca()
    from matplotlib import rc
    rc('font', **fontprop)
    if max(Ns)/1000 % 7 == 0:
        num_ticks = 7
    else:
        num_ticks = 5
    ax.set_xticks(np.linspace(0, max(Ns), num_ticks+1))

    ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontprop)
    ax.set_yticklabels(ax.get_yticks(), fontprop)

    plt.tight_layout()
    plt.savefig('bias_' + bias + '_plot_pvals.pdf', bbox_inches='tight')
    plt.cla()
    plt.close('all')


def pa(bias):
    '''Plot all.'''
    p(bias)
    pm(bias)
    pf(bias)
