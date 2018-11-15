from numpy import (
    cos, array, linspace, arange, loadtxt, vectorize, pi, exp, empty, log, log2,
    finfo, true_divide
    )
from matplotlib import rcParams, rc
from matplotlib.style import use
from matplotlib.pyplot import subplots, close
from matplotlib.cm import get_cmap
from os import getcwd

use('seaborn-paper')
rc('text', usetex=True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath}", r"\usepackage{lmodern}", r"\usepackage{siunitx}", r"\usepackage{units}"
]

def landscape(Ax, Axy, Ay, position1, position2):
    return 0.5*(
        Ax*(1-cos((3*position1)-(2*pi/3)))
        + Axy*(1-cos(position1-position2))
        + Ay*(1-cos((3*position2)))
    )

def plot_equilibrium():

    N = 20
    dx = (2*pi/N)
    amplitudes_x = array([1.0, 2.0, 4.0, 8.0])
    amplitudes_y = array([1.0, 2.0, 4.0, 8.0])
    amplitudes_xy = array([0.0, 1.0, 2.0, 4.0])
    position = linspace(0.0, 2*pi-dx, N)

    for Axy in amplitudes_xy:
        fig, ax = subplots(
            amplitudes_x.size, amplitudes_y.size,
            figsize=(10, 10), sharex='all', sharey='all'
            )
        fig2, ax2 = subplots(
            amplitudes_x.size, amplitudes_y.size,
            figsize=(10, 10), sharex='all', sharey='all'
            )
        for index1, Ax in enumerate(amplitudes_x):
            for index2, Ay in enumerate(amplitudes_y):

                p_eq = empty((N, N), dtype='float64')
                Uprofile = empty((N, N), dtype='float64')
                s = 0.0

                for i in range(N):
                    for j in range(N):
                        Utot = landscape(Ax, Axy, Ay, i*dx, j*dx)
                        s += exp((-1)*Utot)
                        p_eq[i, j] = exp((-1)*Utot)
                        Uprofile[i, j] = Utot

                p_eq /= s

                ax[index1, index2].contourf(position, position, p_eq.T, 20)
                ax2[index1, index2].contourf(position, position, Uprofile.T, 20)

                if index1 == 0:
                    ax[index1, index2].set_title(r"$\mathrm{A}_{x}$ = " + "{0}".format(Ay))
                    ax2[index1, index2].set_title(r"$\mathrm{A}_{x}$ = " + "{0}".format(Ay))

                if index2 == amplitudes_y.size - 1:
                    ax[index1, index2].set_ylabel(r'$\mathrm{A}_{y}$ = ' + "{0}".format(Ax))
                    ax[index1, index2].yaxis.set_label_position("right")
                    ax2[index1, index2].set_ylabel(r'$\mathrm{A}_{y}$ = ' + "{0}".format(Ax))
                    ax2[index1, index2].yaxis.set_label_position("right")

        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig('Axy{0}_N{1}_equilibrium_distr.pdf'.format(Axy, N))
        fig2.savefig('Axy{0}_N{1}_energy_profile.pdf'.format(Axy, N))
        close('all')


def analyze_alex_cython_simulation():

    periods_to_run = 2**(arange(-8.0, 8.0))
    amplitudes = array([0.0, 2.0, 4.0, 8.0])
    trap_strengths = array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

    cycles = 1.0

    fig, ax = subplots(4, 4, figsize=(10, 10))
    target_dir = './output_dir/'
    for row_index in range(4):
        for col_index, A in enumerate(amplitudes):
            for k in trap_strengths:
                filename = '/output_file_k{0}_A{1}_final.dat'.format(k, A)
                times, rot_rate, works, fluxes = loadtxt(
                    target_dir + filename, unpack=True
                    )

                if row_index == 0:
                    quantity_to_plot = fluxes/cycles
                elif row_index == 1:
                    quantity_to_plot = fluxes / (times*cycles)
                elif row_index == 2:
                    quantity_to_plot = (works/cycles)
                elif row_index == 3:
                    quantity_to_plot = works / fluxes
                else:
                    print("index completely out of bounds. WTF?")
                    exit(1)

                ax[row_index, col_index].loglog(
                    rot_rate[::-1], quantity_to_plot[::-1], 'o-', lw=3.0
                )

    fig.tight_layout()
    fig.savefig('likeAlex.pdf')

def analyze_joint_equilibrium_estimate():

    Ax = Ay = Axy = 1.0
    cwd = getcwd()
    pi_est = loadtxt(cwd + '/joint_distribution2.dat')
    N = pi_est.shape[0]
    dx = (2*pi)/N
    positions = linspace(0.0, (2*pi)-dx, N)
    lanscape_vec = vectorize(landscape)
    E = landscape(Ax, Axy, Ay, positions[:,None], positions[None,:])

    pi_theory = exp(-E) / exp(-E).sum(axis=None)

    compare = (pi_est - pi_theory) / pi_theory

    print("Total Variation Distance =", 0.5*((pi_est-pi_theory).__abs__().sum()))
    print("Relative Entropy =", pi_est.dot(log2(true_divide(pi_est, pi_theory))).sum(axis=None))

    # fig, ax = subplots(3, 1, figsize=(10,10), sharex='all', sharey='all')
    # blues_cmap = get_cmap('Blues')
    # cl0 = ax[0].contourf(positions * (180./pi), positions * (180./pi), pi_est.T, 50, cmap=blues_cmap)
    # ax[0].set_title(r'$\hat{\pi}^{\mathrm{eq}}$', fontsize=28)
    # ax[0].set_ylim([0, 360-(360/N)+0.001])
    # oranges_cmap = get_cmap('Oranges')
    # cl1 = ax[1].contourf(positions * (180./pi), positions * (180./pi), pi_theory.T, 50, cmap=oranges_cmap)
    # ax[1].set_title(r'$\pi^{\mathrm{eq}}$', fontsize=28)
    # ax[1].set_ylim([0, 360-(360/N)+0.001])
    # coolwarm_cmap = get_cmap('coolwarm')
    # cl2 = ax[2].contourf(positions * (180./pi), positions * (180./pi), ((pi_est - pi_theory) / pi_theory).T , 50, cmap=coolwarm_cmap)
    # ax[2].set_title(
    #     r'$\left(\hat{\pi}^{\mathrm{eq}}-\pi^{\mathrm{eq}}\right)/\pi^{\mathrm{eq}}$',
    #     fontsize=28
    #     )
    # ax[2].set_ylim([0, 360-(360/N)+0.001])
    # ax[0].tick_params(
    #     axis='y', labelsize=22
    #     )
    # ax[1].tick_params(
    #     axis='y', labelsize=22
    #     )
    # ax[2].tick_params(
    #     axis='both', labelsize=22
    #     )
    # fig.colorbar(cl0, ax=ax[0])
    # fig.colorbar(cl1, ax=ax[1])
    # fig.colorbar(cl2, ax=ax[2])

    # fig.tight_layout()
    # fig.text(
    #     0.03, 0.51, r'$\theta_{y}$', va='center', ha='center',
    #     fontsize=27, rotation='vertical'
    #     )
    # fig.text(
    #     0.47, 0.02, r'$\theta_{x}$', va='center', ha='center',
    #     fontsize=27
    #     )

    # left = 0.1  # the left side of the subplots of the figure
    # right = 1.0    # the right side of the subplots of the figure
    # bottom = 0.09   # the bottom of the subplots of the figure
    # top = 0.95      # the top of the subplots of the figure
    # wspace = 0.1  # the amount of width reserved for blank space between subplots
    # hspace = 0.20  # the amount of height reserved for white space between subplots

    # fig.subplots_adjust(
    #     left=left, right=right, bottom=bottom, top=top,
    #     wspace=wspace, hspace=hspace
    #     )

    # fig.savefig('observe.pdf')
    # close(fig)

    # fig2, ax2 = subplots(1, 1, figsize=(10, 10), sharex='all', sharey='all')

    # bwr_cmap = get_cmap('bwr')
    # cl0 = ax2.contourf(positions * (180./pi), positions *
    #                      (180./pi), -(log(pi_est) / E).T, 50, cmap=bwr_cmap)
    # ax2.set_title('Boltzmann Inversion', fontsize=28)
    # ax2.set_ylim([0, 360-(360/N)+0.001])
    # ax2.set_yticks(arange(0.0, 360.0, 60.0))
    # ax2.set_xticks(arange(0.0, 360.0, 60.0))
    # ax2.tick_params(
    #     axis='both', labelsize=22
    # )
    # cbar = fig2.colorbar(cl0, ax=ax2, ticks=arange(0, 76, 5))
    # cbar.ax.set_ylabel(r'$\beta(x,y)$', fontsize=20)

    # fig2.tight_layout()
    # fig2.text(
    #     0.03, 0.51, r'$\theta_{y}$', va='center', ha='center',
    #     fontsize=27, rotation='vertical'
    # )
    # fig2.text(
    #     0.47, 0.02, r'$\theta_{x}$', va='center', ha='center',
    #     fontsize=27
    # )

    # left = 0.13  # the left side of the subplots of the figure
    # right = 0.98    # the right side of the subplots of the figure
    # bottom = 0.09   # the bottom of the subplots of the figure
    # top = 0.95      # the top of the subplots of the figure
    # wspace = 0.1  # the amount of width reserved for blank space between subplots
    # hspace = 0.20  # the amount of height reserved for white space between subplots

    # fig2.subplots_adjust(
    #     left=left, right=right, bottom=bottom, top=top,
    #     wspace=wspace, hspace=hspace
    # )

    # fig2.savefig('observe2.pdf')
    # close('all')


if __name__ == "__main__":
    # analyze_alex_cython_simulation()
    # plot_equilibrium()
    analyze_joint_equilibrium_estimate()
