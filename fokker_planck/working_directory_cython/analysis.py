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
    r"\usepackage{amsmath}", r"\usepackage{lmodern}",
    r"\usepackage{siunitx}", r"\usepackage{units}",
    r"\usepackage{physics}", r"\usepackage{bm}"
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
                        Utot = landscape(Ax, Axy, Ay, position[i], position[j])
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
    pi_est = loadtxt(cwd + '/joint_distribution3.dat')
    N = pi_est.shape[0]
    dx = (2*pi)/N
    positions = linspace(0.0, (2*pi)-dx, N)
    landscape_vec = vectorize(landscape)
    E = landscape_vec(Ax, Axy, Ay, positions[:,None], positions[None,:])

    Z = exp((-1.0)*E).sum(axis=None)
    F = -log(Z)
    V_theory = E - F
    V_est = -log(pi_est)
    pi_theory = exp((-1.0)*E) / exp((-1.0)*E).sum(axis=None)


    compare = (pi_est - pi_theory) / pi_theory

    print("Total Variation Distance =", 0.5*((pi_est-pi_theory).__abs__().sum()))
    print("Relative Entropy =", pi_est.dot(log2(true_divide(pi_est, pi_theory))).sum(axis=None))

    fig, ax = subplots(3, 1, figsize=(10,10), sharex='all', sharey='all')
    blues_cmap = get_cmap('Blues')
    cl0 = ax[0].contourf(positions * (180./pi), positions * (180./pi), pi_est.T, cmap=blues_cmap)
    ax[0].set_title(r'$\hat{\pi}^{\mathrm{eq}}$', fontsize=28)
    ax[0].set_ylim([0, 360-(360/N)+0.001])
    oranges_cmap = get_cmap('Oranges')
    cl1 = ax[1].contourf(positions * (180./pi), positions * (180./pi), pi_theory.T, cmap=oranges_cmap)
    ax[1].set_title(r'$\pi^{\mathrm{eq}}$', fontsize=28)
    ax[1].set_ylim([0, 360-(360/N)+0.001])
    coolwarm_cmap = get_cmap('coolwarm')
    cl2 = ax[2].contourf(positions * (180./pi), positions * (180./pi), ((pi_est - pi_theory) / pi_theory).T , cmap=coolwarm_cmap)
    ax[2].set_title(
        r'$\left(\hat{\pi}^{\mathrm{eq}}-\pi^{\mathrm{eq}}\right)/\pi^{\mathrm{eq}}$',
        fontsize=28
        )
    ax[2].set_ylim([0, 360-(360/N)+0.001])
    ax[0].tick_params(
        axis='y', labelsize=22
        )
    ax[1].tick_params(
        axis='y', labelsize=22
        )
    ax[2].tick_params(
        axis='both', labelsize=22
        )
    fig.colorbar(cl0, ax=ax[0])
    fig.colorbar(cl1, ax=ax[1])
    fig.colorbar(cl2, ax=ax[2])

    fig.tight_layout()
    fig.text(
        0.03, 0.51, r'$\theta_{y}$', va='center', ha='center',
        fontsize=27, rotation='vertical'
        )
    fig.text(
        0.47, 0.02, r'$\theta_{x}$', va='center', ha='center',
        fontsize=27
        )

    left = 0.1  # the left side of the subplots of the figure
    right = 1.0    # the right side of the subplots of the figure
    bottom = 0.09   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.1  # the amount of width reserved for blank space between subplots
    hspace = 0.20  # the amount of height reserved for white space between subplots

    fig.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top,
        wspace=wspace, hspace=hspace
        )

    fig.savefig('observe.pdf')

    # fig2, ax2 = subplots(2, 1, figsize=(10, 10), sharex='all', sharey='all')

    # bwr_cmap = get_cmap('bwr')
    # cl0 = ax2.contourf(positions * (180./pi), positions *
    #                      (180./pi), -(log(pi_est)).T, 50, cmap=bwr_cmap)
    # cl0 = ax2.contourf(positions * (180./pi), positions *
    #                      (180./pi), -(log(pi_est)).T, 50, cmap=bwr_cmap)

    # ax2.set_title('Boltzmann Inversion', fontsize=28)
    # ax2.set_ylim([0, 360-(360/N)+0.001])
    # ax2.set_yticks(arange(0.0, 360.0, 60.0))
    # ax2.set_xticks(arange(0.0, 360.0, 60.0))
    # ax2.tick_params(
    #     axis='both', labelsize=22
    # )
    # cbar = fig2.colorbar(cl0, ax=ax2)
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

    fig3, ax3 = subplots(3, 1, figsize=(10, 10), sharex='all', sharey='all')
    blues_cmap = get_cmap('Blues')
    cl0_3 = ax3[0].contourf(positions * (180./pi), positions *
                         (180./pi), V_theory, cmap=blues_cmap)
    ax3[0].set_title(r'$\hat{V}$', fontsize=28)
    ax3[0].set_ylim([0, 360-(360/N)+0.001])
    oranges_cmap = get_cmap('Oranges')
    cl1_3 = ax3[1].contourf(positions * (180./pi), positions *
                         (180./pi), V_est, cmap=oranges_cmap)
    ax3[1].set_title(r'$V_{\mathrm{theory}}$', fontsize=28)
    ax3[1].set_ylim([0, 360-(360/N)+0.001])
    coolwarm_cmap = get_cmap('coolwarm')
    cl2_3 = ax3[2].contourf(positions * (180./pi), positions * (180./pi),
                         (V_est-V_theory)/V_theory, cmap=coolwarm_cmap)
    ax3[2].set_title(
        r'$\left(\hat{V}-V_{\mathrm{theory}}\right)/V_{\mathrm{theory}}$',
        fontsize=28
    )
    ax3[2].set_ylim([0, 360-(360/N)+0.001])
    ax3[0].tick_params(
        axis='y', labelsize=22
    )
    ax3[1].tick_params(
        axis='y', labelsize=22
    )
    ax3[2].tick_params(
        axis='both', labelsize=22
    )
    fig3.colorbar(cl0_3, ax=ax3[0])
    fig3.colorbar(cl1_3, ax=ax3[1])
    fig3.colorbar(cl2_3, ax=ax3[2])

    fig3.tight_layout()
    fig3.text(
        0.03, 0.51, r'$\theta_{y}$', va='center', ha='center',
        fontsize=27, rotation='vertical'
    )
    fig3.text(
        0.47, 0.02, r'$\theta_{x}$', va='center', ha='center',
        fontsize=27
    )

    left = 0.1  # the left side of the subplots of the figure
    right = 1.0    # the right side of the subplots of the figure
    bottom = 0.09   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.1  # the amount of width reserved for blank space between subplots
    hspace = 0.20  # the amount of height reserved for white space between subplots

    fig3.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top,
        wspace=wspace, hspace=hspace
    )

    fig3.savefig('observe3.pdf')

def analyze():

    ID = 3
    N = 360
    Ax = 10.0
    Axy = Ay = A = 0.0
    H = 3.0

    cwd = getcwd()
    target_dir = cwd + '/output_dir/'
    data_filename = '/ID_{0}_outfile.dat'.format(ID)
    data_total_path = target_dir + data_filename

    p_now, p_equil, flux1, flux2 = loadtxt(data_total_path, unpack=True)

    p_now = p_now.reshape((N, N))
    p_equil = p_equil.reshape((N, N))
    flux1 = flux1.reshape((N, N))
    flux2 = flux2.reshape((N, N))

    positions = linspace(0.0, 2*pi-(2*pi/N), N) * (180/pi)

    fig, ax = subplots(2, 2, figsize=(10, 10), sharex='all', sharey='all')
    cl0 = ax[0, 0].contourf(positions, positions, p_now.T, cmap=get_cmap('Reds'))
    cl1 = ax[0, 1].contourf(positions, positions, p_equil.T, cmap=get_cmap('Blues'))
    cl2 = ax[1, 0].contourf(positions, positions, flux1.T, cmap=get_cmap('coolwarm'))
    cl3 = ax[1, 1].contourf(positions, positions, flux2.T, cmap=get_cmap('bwr'))

    titles = {
        (0,0): r'$\hat{p}^{\mathrm{SS}}(\bm{\theta})$',
        (0,1): r'$\pi(\bm{\theta})$',
        (1,0): r'$\vb{J}_{X}^{\mathrm{SS}}(\bm{\theta})$',
        (1,1): r'$\vb{J}_{Y}^{\mathrm{SS}}(\bm{\theta})$'
        }

    for i in range(2):
        for j in range(2):
            ax[i, j].set_title(titles[(i, j)], fontsize=20)
            ax[i, j].set_xticks(array([0.0, 2*pi/9, 8*pi/9, 14*pi/9])*(180/pi))
            ax[i, j].set_xticklabels(
                [
                    r'$0$', r'$\frac{2\pi}{9}$', r'$\frac{8\pi}{9}$',
                    r'$\frac{14\pi}{9}$'
                    ]
                    )
            ax[i, j].set_yticks(arange(0, 360, 60))
            ax[i, j].tick_params(axis='both', labelsize=18)

    fig.colorbar(cl0, ax=ax[0, 0])
    fig.colorbar(cl1, ax=ax[0, 1])
    fig.colorbar(cl2, ax=ax[1, 0])
    fig.colorbar(cl3, ax=ax[1, 1])

    fig.tight_layout()

    fig.text(
        0.5, 0.98,
        'Ax = {0}, Axy = {1}, Ay = {2}, H = {3}, A = {4}'.format(Ax, Axy, Ay, H, A),
        fontsize=28, va='center', ha='center'
        )
    fig.text(
        0.5, 0.03,
        r'$\theta_{X}$',
        fontsize=28, va='center', ha='center'
        )
    fig.text(
        0.03, 0.51,
        r'$\theta_{Y}$',
        fontsize=28, va='center', ha='center', rotation='vertical'
        )

    left = 0.09  # the left side of the subplots of the figure
    # right = 1.0    # the right side of the subplots of the figure
    bottom = 0.09   # the bottom of the subplots of the figure
    top = 0.92      # the top of the subplots of the figure
    # wspace = 0.1  # the amount of width reserved for blank space between subplots
    # hspace = 0.20  # the amount of height reserved for white space between subplots

    fig.subplots_adjust(
        left=left, top=top, bottom=bottom
    )

    fig.savefig(target_dir + 'ID_{}_plot.pdf'.format(ID))

if __name__ == "__main__":
    # analyze_alex_cython_simulation()
    # plot_equilibrium()
    # analyze_joint_equilibrium_estimate()
    analyze()
