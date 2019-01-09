#!/anaconda3/bin/python
from math import cos, sin, pi
from numpy import (
    array, linspace, arange, loadtxt, vectorize, pi as npi, exp, empty, log, log2,
    finfo, true_divide
    )
from matplotlib import rcParams, rc, ticker, colors
from matplotlib.style import use
from matplotlib.pyplot import subplots, close
from matplotlib.cm import get_cmap
from os import getcwd
from datetime import datetime

use('seaborn-paper')
rc('text', usetex=True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath}", r"\usepackage{lmodern}",
    r"\usepackage{siunitx}", r"\usepackage{units}",
    r"\usepackage{physics}", r"\usepackage{bm}"
]

def landscape(Ax, Axy, Ay, position1, position2):
    return 0.5*(Ax*(1-cos((3*position1)-(2*pi/3)))+Axy*(1-cos(position1-position2))+Ay*(1-cos((3*position2))))

def force1(position1, position2, m1, gamma, Ax, Axy, Ay, H):  # force on system X
    return (0.5)*(Axy*sin(position1-position2)+(3*Ax*sin((3*position1)-(2*pi/3)))) + H

def force2(position1, position2, m2, gamma, Ax, Axy, Ay, A):  # force on system Y
    return (0.5)*((-1.0)*Axy*sin(position1-position2)+(3*Ay*sin(3*position2))) - A

def analyze_energetics():

    N = 1080
    dx = (2*pi/N)
    amplitudes_x = array([1.0, 2.0, 4.0, 8.0, 10.0])
    amplitudes_y = array([0.0, 1.0, 2.0, 4.0, 8.0, 10.0])
    amplitudes_xy = array([0.0, 1.0, 2.0, 4.0])
    # amplitudes_x = array([10.0])
    # amplitudes_y = array([0.0])
    # amplitudes_xy = array([0.0])
    position = linspace(0.0, 2*pi-dx, N)

    vec_landscape = vectorize(landscape)
    vec_force1 = vectorize(force1)
    vec_force2 = vectorize(force2)
    gamma = 1000
    m1 = m2 = 1.0
    H = 10
    A = 0

    for Axy in amplitudes_xy:
        fig, ax = subplots(
            amplitudes_x.size, amplitudes_y.size,
            figsize=(10, 10), sharex='all', sharey='all'
            )
        fig2, ax2 = subplots(
            amplitudes_x.size, amplitudes_y.size,
            figsize=(10, 10), sharex='all', sharey='all'
            )
        fig3, ax3 = subplots(
            amplitudes_x.size, amplitudes_y.size,
            figsize=(10, 10), sharex='all', sharey='all'
            )
        fig4, ax4 = subplots(
            amplitudes_x.size, amplitudes_y.size,
            figsize=(10, 10), sharex='all', sharey='all'
            )
        for index1, Ax in enumerate(amplitudes_x):
            for index2, Ay in enumerate(amplitudes_y):

                Utot = vec_landscape(Ax, Axy, Ay, position[:, None], position[None, :])
                p_eq = exp((-1.0)*Utot) / exp((-1.0)*Utot).sum(axis=None)
                force1_array = vec_force1(position[:, None], position[None, :], m1, gamma, Ax, Axy, Ay, H)
                force2_array = vec_force2(position[:, None], position[None, :], m2, gamma, Ax, Axy, Ay, A)

                ax[index1, index2].contourf(position, position, p_eq.T)
                ax2[index1, index2].contourf(position, position, Utot.T)
                ax3[index1, index2].contourf(position, position, force1_array.T)
                ax4[index1, index2].contourf(position, position, force2_array.T)

                if index1 == 0:
                    ax[index1, index2].set_title(r"$\mathrm{A}_{x}$ = " + "{0}".format(Ay))
                    ax2[index1, index2].set_title(r"$\mathrm{A}_{x}$ = " + "{0}".format(Ay))
                    ax3[index1, index2].set_title(r"$\mathrm{A}_{x}$ = " + "{0}".format(Ay))
                    ax4[index1, index2].set_title(r"$\mathrm{A}_{x}$ = " + "{0}".format(Ay))

                if index2 == amplitudes_y.size - 1:
                    ax[index1, index2].set_ylabel(r'$\mathrm{A}_{y}$ = ' + "{0}".format(Ax))
                    ax[index1, index2].yaxis.set_label_position("right")
                    ax2[index1, index2].set_ylabel(r'$\mathrm{A}_{y}$ = ' + "{0}".format(Ax))
                    ax2[index1, index2].yaxis.set_label_position("right")
                    ax3[index1, index2].set_ylabel(r'$\mathrm{A}_{y}$ = ' + "{0}".format(Ax))
                    ax3[index1, index2].yaxis.set_label_position("right")
                    ax4[index1, index2].set_ylabel(r'$\mathrm{A}_{y}$ = ' + "{0}".format(Ax))
                    ax4[index1, index2].yaxis.set_label_position("right")

        fig.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        fig.savefig('Axy{0}_N{1}_equilibrium_distr.pdf'.format(Axy, N))
        fig2.savefig('Axy{0}_N{1}_energy_profile.pdf'.format(Axy, N))
        fig3.savefig('Axy{0}_N{1}_force1_profile.pdf'.format(Axy, N))
        fig4.savefig('Axy{0}_N{1}_force2_profile.pdf'.format(Axy, N))
        close('all')

def analyze():

    ID = 1
    N = 360
    Ax = 4.0
    Axy = 0.0
    Ay = 4.0
    H = 0.0
    A = 0.0

    cwd = getcwd()
    target_dir = cwd + '/output_dir/'
    #data_filename = '/ID_{0}_outfile.dat'.format(ID)
    data_filename = 'Ax_{0}_Axy_{1}_Ay_{2}_Fh_{3}_Fa_{4}_outfile.dat'.format(
        Ax, Axy, Ay, H, A)
    data_total_path = target_dir + data_filename

    print("{} Loading data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    p_now, p_equil, flux1, flux2, potential, force1, force2 = loadtxt(data_total_path, unpack=True)
    # p_now,  flux1, force1 = loadtxt(data_total_path, unpack=True, usecols=(0, 2, 5))

    p_now = p_now.reshape((N, N))
    p_equil = p_equil.reshape((N, N))
    flux1 = flux1.reshape((N, N))
    flux2 = flux2.reshape((N, N))
    potential = potential.reshape((N, N))
    force1 = force1.reshape((N, N))
    force2 = force2.reshape((N, N))

    positions = linspace(0.0, 2*pi-(2*pi/N), N)*(180/pi)

    to_plot = [p_now, p_equil, flux1, flux2]
    vmin0 = min([array_to_plot.min() for array_to_plot in to_plot[:2]])
    vmax0 = max([array_to_plot.max() for array_to_plot in to_plot[:2]])
    vmin1 = min([array_to_plot.min() for array_to_plot in to_plot[2:]])
    vmax1 = max([array_to_plot.max() for array_to_plot in to_plot[2:]])
    cmap0=get_cmap('gnuplot2')
    cmap1=get_cmap('coolwarm')

    print("{} Plotting the contours...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    fig, ax = subplots(2, 2, figsize=(10, 10), sharex='all', sharey='all')
    cl0 = ax[0, 0].contourf(positions, positions, p_now.T, 30, vmin=vmin0, vmax=vmax0, cmap=cmap0)
    cl1 = ax[0, 1].contourf(positions, positions, p_equil.T, 30, vmin=vmin0, vmax=vmax0, cmap=cmap0)
    cl2 = ax[1, 0].contourf(positions, positions, flux1.T, 30, vmin=vmin1, vmax=vmax1, cmap=cmap1)
    cl3 = ax[1, 1].contourf(positions, positions, flux2.T, 30, vmin=vmin1, vmax=vmax1, cmap=cmap1)

    titles = {
        (0,0): r'$\hat{p}^{\mathrm{SS}}(\bm{\theta})$',
        (0,1): r'$\pi(\bm{\theta})$',
        (1,0): r'$\vb{J}_{X}^{\mathrm{SS}}(\bm{\theta})$',
        (1,1): r'$\vb{J}_{Y}^{\mathrm{SS}}(\bm{\theta})$'
        }

    print("{} Making the plot look nice...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
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

    sfmt=ticker.ScalarFormatter()
    sfmt.set_powerlimits((0, 0))
    fig.colorbar(cl0, ax=ax[0, 0], format=sfmt)
    fig.colorbar(cl1, ax=ax[0, 1], format=sfmt)
    fig.colorbar(cl2, ax=ax[1, 0], format=sfmt)
    fig.colorbar(cl3, ax=ax[1, 1], format=sfmt)

    fig.tight_layout()

    title_str = (
        r'$A_{x}$ = ' + str(Ax)
        + r', $A_{xy}$ = ' + str(Axy)
        + r', $A_{y}$ = ' + str(Ay)
        + r', $F_{X}$ = ' + str(H)
        + r', $F_{Y}$ = '  + str(A)
        )

    fig.text(
        0.5, 0.975,
        title_str,
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

    print("{} Saving the plot...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    fig.savefig(target_dir + 'ID_{}_plot.pdf'.format(ID))
    print("{} Done!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    fig2, ax2 = subplots(3,1, sharex=True, figsize=(10,10))
    ax2[0].plot(positions, flux1.T[N//4, :], "X-", lw=3.0, ms=8)
    ax2[0].set_ylabel(titles[(1, 0)], fontsize=18)
    ax2[0].set_ylim([1.5*vmin1, 1.5*vmax1])
    ax2[1].plot(positions, flux2.T[N//4, :], "X-", lw=3.0, ms=8)
    ax2[1].set_ylabel(titles[(1, 1)], fontsize=18)
    ax2[1].set_ylim([1.5*vmin1, 1.5*vmax1])
    ax2[2].plot(positions, p_now.T[N//4, :], "go-", lw=3.0, ms=8)
    ax2[2].set_ylim(bottom=0.0)
    ax2[2].set_ylabel(titles[(0, 0)], fontsize=18)
    ax2[2].set_xlabel(r"$\theta_{X}$", fontsize=18)

    for i in range(3):
        ax2[i].grid(True)
        ax2[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax2[i].yaxis.offsetText.set_fontsize(14)

    fig2.tight_layout()
    fig2.savefig("slice-cluster.pdf")

    fig3, ax3 = subplots(3,1, sharex=True, figsize=(10,10))
    ax3[0].plot(positions, potential[:, N//2], lw=3.0, ms=8)
    ax3[0].set_ylabel(r"$V_{\mathrm{tot}}$", fontsize=18)
    ax3[1].plot(positions, force1[:, N//2], lw=3.0, ms=8)
    ax3[1].set_ylabel(r"$F_{H}$", fontsize=18)
    ax3[2].plot(positions, force2[:, N//2], lw=3.0, ms=8)
    ax3[2].set_ylabel(r"$F_{A}$", fontsize=18)
    ax3[2].set_xlabel(r"$\theta_{X}$", fontsize=18)
    ax3[0].grid(True)
    ax3[1].grid(True)
    ax3[2].grid(True)

    fig3.tight_layout()
    fig3.savefig("slice_energetics-cluster.pdf")

    close("all")


def analyze2():

    N = 360

    cwd = getcwd()
    target_dir = cwd + '/output_dir/'
    #data_filename = '/ID_{0}_outfile.dat'.format(ID)
    data_filename = 'Ax_{0}_Axy_{1}_Ay_{2}_Fh_{3}_Fa_{4}_outfile.dat'.format(
        Ax, Axy, Ay, H, A)
    data_total_path = target_dir + data_filename

    print("{} Loading data...".format(
        datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    p_now, p_equil, flux1, flux2, potential, force1, force2 = loadtxt(
        data_total_path, unpack=True)
    # p_now,  flux1, force1 = loadtxt(data_total_path, unpack=True, usecols=(0, 2, 5))

    p_now = p_now.reshape((N, N))
    p_equil = p_equil.reshape((N, N))
    flux1 = flux1.reshape((N, N))
    flux2 = flux2.reshape((N, N))
    potential = potential.reshape((N, N))
    force1 = force1.reshape((N, N))
    force2 = force2.reshape((N, N))

    positions = linspace(0.0, 2*pi-(2*pi/N), N)*(180/pi)




if __name__ == "__main__":
    # analyze_energetics()
    analyze()
