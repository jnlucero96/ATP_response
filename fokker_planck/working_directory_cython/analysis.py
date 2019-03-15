#!/anaconda3/bin/python
from math import cos, sin, pi
from numpy import (
    array, linspace, arange, loadtxt, vectorize, pi as npi, exp,
    empty, log, log2, finfo, true_divide, asarray
    )
from scipy.integrate import trapz
from matplotlib import rcParams, rc, ticker, colors, cm
from matplotlib.style import use
from matplotlib.pyplot import subplots, close
from matplotlib.cm import get_cmap
from os import getcwd
from datetime import datetime

from utilities import calc_flux

use('seaborn-paper')
rc('text', usetex=True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath}", r"\usepackage{lmodern}",
    r"\usepackage{siunitx}", r"\usepackage{units}",
    r"\usepackage{physics}", r"\usepackage{bm}"
]

def set_params():

    N = 360
    E0 = 4.0
    Ecouple = 8.0
    E1 = 4.0
    F_Hplus = 0.0
    F_atp = -2.0
    num_minima = 9.0
    phase_shift = 0.0

    m1 = 1.0
    m2 = 1.0
    beta = 1.0
    gamma = 1000.0

    return (
        N, E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift,
        m1, m2, beta, gamma
        )

# ============================================================================
# ==============
# ============== ENERGY LANDSCAPES
# ==============
# ============================================================================

def landscape(E0, Ecouple, E1, num_minima, phase_shift, position1, position2):
    return 0.5*(
        E0*(1-cos(num_minima*position1-phase_shift))
        +Ecouple*(1-cos(position1-position2))
        +E1*(1-cos((num_minima*position2)))
        )

def force1(E0, Ecouple, F_Hplus, num_minima, phase_shift, position1, position2):  # force on system X
    return (0.5)*(
        Ecouple*sin(position1-position2)
        +(num_minima*E0*sin(num_minima*position1-phase_shift))
        ) - F_Hplus

def force2(E1, Ecouple, F_atp, num_minima, position1, position2):  # force on system Y
    return (0.5)*(
        (-1.0)*Ecouple*sin(position1-position2)+(num_minima*E1*sin(num_minima*position2))
        ) - F_atp

# ============================================================================
# ==============
# ============== MAJOR CALCULATIONS
# ==============
# ============================================================================

def calculate_flux_power_and_efficiency(target_dir=None):

    input_file_name = (
        "/reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )
    output_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, E0, __, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    F_Hplus_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    F_atp_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])

    dx = (2*pi)/N

    positions = linspace(0.0,2*pi-dx,N)

    for F_atp in F_atp_array:
        for F_Hplus in F_Hplus_array:
            integrate_flux_X = empty(Ecouple_array.size)
            integrate_flux_Y = empty(Ecouple_array.size)
            integrate_power_X = empty(Ecouple_array.size)
            integrate_power_Y = empty(Ecouple_array.size)
            efficiency_ratio = empty(Ecouple_array.size)

            with open(
                target_dir + output_file_name.format(
                    E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                    ), "w"
                    ) as ofile:

                for ii, Ecouple in enumerate(Ecouple_array):

                    flux_array = empty((2,N,N))
                    prob_ss_array, __, __, force1_array, force2_array = loadtxt(
                        target_dir + input_file_name.format(
                            E0,Ecouple,E1,F_Hplus,F_atp,num_minima,phase_shift
                            ), unpack=True
                    )

                    prob_ss_array = prob_ss_array.reshape((N,N))
                    force1_array = force1_array.reshape((N,N))
                    force2_array = force2_array.reshape((N,N))

                    calc_flux(
                        positions, prob_ss_array, force1_array, force2_array,
                        flux_array, m1, m2, gamma, beta, N, dx, 0.001
                        )

                    flux_array = asarray(flux_array)
                    force1_array = asarray(force1_array)
                    force2_array = asarray(force2_array)

                    integrate_flux_X[ii] = trapz(
                        trapz(flux_array[0],axis=1, dx=dx), dx=dx
                        )/(2*pi) # integrate over Y then average over X
                    integrate_flux_Y[ii] = trapz(
                        trapz(flux_array[1],axis=0, dx=dx), dx=dx
                        )/(2*pi) # integrate over X then average over Y
                    integrate_power_X[ii] = trapz(
                        trapz(flux_array[0]*F_Hplus,axis=1, dx=dx), dx=dx
                        )/(2*pi) # integrate over Y then average over X
                    integrate_power_Y[ii] = trapz(
                        trapz(flux_array[1]*F_atp,axis=0, dx=dx), dx=dx
                        )/(2*pi) # integrate over X then average over Y

                if (abs(F_Hplus) <= abs(F_atp)):
                    efficiency_ratio = -(integrate_power_X/integrate_power_Y)
                else:
                    efficiency_ratio = -(integrate_power_Y/integrate_power_X)

                for ii, Ecouple in enumerate(Ecouple_array):

                    ofile.write(
                        "{0:.15e}".format(Ecouple) + "\t"
                        + "{0:.15e}".format(integrate_flux_X[ii]) + "\t"
                        + "{0:.15e}".format(integrate_flux_Y[ii]) + "\t"
                        + "{0:.15e}".format(integrate_power_X[ii]) + "\t"
                        + "{0:.15e}".format(integrate_power_Y[ii]) + "\t"
                        + "{0:.15e}".format(efficiency_ratio[ii]) + "\n"
                    )

# ============================================================================
# ==============
# ============== SINGLE PLOTS
# ==============
# ============================================================================

def plot_energy():

    [
        N, E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = 2*pi/N

    positions = linspace(0.0, 2*pi-dx, N)
    positions_deg = positions * (180/pi)

    vec_landscape = vectorize(landscape)

    V = vec_landscape(
        E0, Ecouple, E1, num_minima, phase_shift,
        positions[:,None], positions[None,:]
        )

    fig, ax = subplots(1, 1, figsize=(10,10))

    ct = ax.contourf(
        positions_deg, positions_deg, V.T, 30, cmap=cm.get_cmap("afmhot")
        )
    ax.set_ylabel(r"$\theta_{\mathrm{F}_{1}}$", fontsize=20)
    ax.set_xlabel(r"$\theta_{\mathrm{F}_{\mathrm{o}}}$", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xticks(arange(0.0, 361, 60))
    ax.set_yticks(arange(0.0, 361, 60))

    fig.colorbar(ct, ax=ax)
    fig.tight_layout()
    fig.savefig(
        "energy_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_E_Hplus_{3}_E_atp_{4}".format(
            E0, Ecouple, E1, F_Hplus, F_atp
            ) + "_figure.pdf"
        )

def plot_probability(target_dir):

    [
        N, E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = 2*pi/N

    input_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}".format(
            E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
        ) + "_outfile.dat")
    prob_array = loadtxt(
        target_dir + input_file_name,
        usecols=(0,)
        ).reshape((N, N));
    prob_eq_array = loadtxt(
        target_dir + input_file_name, usecols=(1,)
        ).reshape((N, N));

    to_plot = [
        prob_array, prob_eq_array
    ]
    vmin = min([array_to_plot.min() for array_to_plot in to_plot])
    vmax = max([array_to_plot.max() for array_to_plot in to_plot])

    positions = linspace(0.0, 2*pi-dx, N)

    fig, ax = subplots(1, 1, figsize=(10,10))

    cl0 = ax.contourf(
        positions, positions, prob_eq_array.T, 30,
        vmin=vmin, vmax=vmax, cmap=cm.get_cmap("gnuplot")
        )
    ax.contour(
        positions, positions, prob_array.T, 30,
        vmin=vmin, vmax=vmax, cmap=cm.get_cmap("cool")
        )

    ax.set_xticks(arange(0, 361, 60))
    ax.set_yticks(arange(0, 361, 60))
    ax.set_ylabel(r"$\theta_{1}$", fontsize=20)
    ax.set_xlabel(r"$\theta_{0}$", fontsize=20)
    ax.tick_params(labelsize=16, axis='both')
    ax.grid(True)

    sfmt=ticker.ScalarFormatter()
    sfmt.set_powerlimits((0, 0))
    fig.colorbar(cl0, ax=ax, format=sfmt)

    fig.tight_layout()
    fig.savefig(
        "probability_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_E_Hplus_{3}_E_atp_{4}".format(
            E0, Ecouple, E1, F_Hplus, F_atp
            ) + "_figure.pdf"
        )

def plot_efficiency(target_dir):

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = (2*pi)/N

    Ecouple_array, efficiency_ratio = loadtxt(
        target_dir + input_file_name.format(
            E0, E1, F_Hplus, F_atp, num_minima, phase_shift
            ),
        unpack=True, usecols=(0,5)
    )

    fig, ax = subplots(1, 1, figsize=(10,10))

    l0 = ax.plot(
        Ecouple_array, efficiency_ratio, lw=3.0
        )

    print(efficiency_ratio)

    ax.set_ylabel(
        r"$-\left(\mathcal{P}_{\mathrm{H}^{+}}/\mathcal{P}_{\mathrm{atp}}\right)$",
        fontsize=20
        )
    ax.set_xlabel(r"$E_{\mathrm{couple}}$", fontsize=20)
    ax.tick_params(labelsize=16, axis='both')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        target_dir + "/efficiency_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_E_Hplus_{3}_E_atp_{4}".format(
            E0, Ecouple, E1, F_Hplus, F_atp
            ) + "_figure.pdf"
        )

# ============================================================================
# ==============
# ============== SCAN PLOTS
# ==============
# ============================================================================

def plot_flux_scan(target_dir):

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, E0, __, E1, F_Hplus, F_atp, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = (2*pi)/N

    # F_Hplus_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    # F_atp_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    F_Hplus_array = array([0.0, 2.0, 4.0, 8.0])
    F_atp_array = array([-8.0, -4.0, -2.0, 0.0])[::-1]

    fig, ax = subplots(
        F_Hplus_array.size, F_atp_array.size,
        figsize=(15,15), sharex='col', sharey='all'
        )

    positions = linspace(0.0, 2*pi-dx, N)

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):
            Ecouple_array, integrate_flux_X, integrate_flux_Y = loadtxt(
                target_dir + input_file_name.format(
                    E0,E1,F_Hplus,F_atp,num_minima,phase_shift
                    ),
                unpack=True, usecols=(0,1,2)
            )

            ax[row_index, col_index].plot(
                Ecouple_array, integrate_flux_X,
                lw=3.0, label=r"$J_{\mathrm{H}^{+}}$"
                )
            ax[row_index, col_index].plot(
                Ecouple_array, integrate_flux_Y,
                lw=3.0, label=r"$J_{\mathrm{atp}}$"
                )

            if (row_index == 0):
                ax[row_index, col_index].set_title(
                    "{}".format(F_Hplus), fontsize=20
                    )
            if (col_index == F_Hplus_array.size - 1):
                ax[row_index, col_index].set_ylabel(
                    "{}".format(F_atp), fontsize=20
                    )
                ax[row_index, col_index].yaxis.set_label_position("right")

    for i in range(F_atp_array.size):
        for j in range(F_Hplus_array.size):
            ax[i, j].grid(True)
            ax[i, j].tick_params(labelsize=15)
            ax[i, j].yaxis.offsetText.set_fontsize(12)
            if (i==j):
                ax[i, j].legend(loc=0, prop={"size": 10})
            ax[i, j].ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    fig.text(
        0.03, 0.51, r"$\langle J\rangle$", fontsize=20, rotation="vertical"
        )
    fig.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=20, rotation="vertical"
        )
    fig.text(0.49, 0.03, r"$E_{\mathrm{couple}}$", fontsize=20)
    fig.text(0.50, 0.97, r"$F_{H^{+}}$", fontsize=20)

    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(target_dir + "/flux_scan_figure.pdf")

def plot_power_scan(target_dir):

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, E0, __, E1, __, __, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = (2*pi)/N

    # F_Hplus_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    # F_atp_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    F_Hplus_array = array([0.0, 2.0, 4.0, 8.0])
    F_atp_array = array([-8.0, -4.0, -2.0, 0.0])[::-1]

    fig, ax = subplots(
        F_Hplus_array.size, F_atp_array.size,
        figsize=(15,15), sharex='col', sharey='all'
        )

    positions = linspace(0.0, 2*pi-dx, N)

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):
            Ecouple_array, integrate_power_X, integrate_power_Y = loadtxt(
                target_dir + input_file_name.format(
                    E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                    ),
                unpack=True, usecols=(0,3,4)
            )

            ax[row_index, col_index].plot(
                Ecouple_array, integrate_power_X,
                lw=3.0, label=r"$\mathcal{P}_{\mathrm{H}^{+}}$"
                )
            ax[row_index, col_index].plot(
                Ecouple_array, integrate_power_Y,
                lw=3.0, label=r"$\mathcal{P}_{\mathrm{atp}}$"
                )

            if (row_index == 0):
                ax[row_index, col_index].set_title(
                    "{}".format(F_Hplus), fontsize=20
                    )
            if (col_index == F_Hplus_array.size - 1):
                ax[row_index, col_index].set_ylabel(
                    "{}".format(F_atp), fontsize=20
                    )
                ax[row_index, col_index].yaxis.set_label_position("right")

    for i in range(F_atp_array.size):
        for j in range(F_Hplus_array.size):
            ax[i, j].grid(True)
            ax[i, j].tick_params(labelsize=15)
            ax[i, j].yaxis.offsetText.set_fontsize(12)
            if (i==j):
                ax[i, j].legend(loc=0, prop={"size": 10})
            ax[i, j].ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    fig.text(
        0.03, 0.51, r"$\mathcal{P}$", fontsize=20, rotation="vertical"
        )
    fig.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=20, rotation="vertical"
        )
    fig.text(0.49, 0.03, r"$E_{\mathrm{couple}}$", fontsize=20)
    fig.text(0.50, 0.97, r"$F_{H^{+}}$", fontsize=20)

    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(target_dir + "/power_scan_figure.pdf")

def plot_efficiency_scan(target_dir):

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, E0, __, E1, __, __, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = (2*pi)/N

    # F_Hplus_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    # F_atp_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    F_Hplus_array = array([0.0, 2.0, 4.0, 8.0])
    F_atp_array = array([-8.0, -4.0, -2.0, 0.0])[::-1]

    fig, ax = subplots(
        F_Hplus_array.size, F_atp_array.size,
        figsize=(15,15), sharex='col', sharey='all'
        )

    positions = linspace(0.0, 2*pi-dx, N)

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

            Ecouple_array, efficiency_ratio = loadtxt(
                target_dir + input_file_name.format(
                    E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                    ),
                unpack=True, usecols=(0,5)
            )

            if (abs(F_Hplus) == abs(F_atp)): continue

            ax[row_index, col_index].plot(
                Ecouple_array, efficiency_ratio, lw=3.0
                )

            # if (row_index == 0):
            #     ax[row_index, col_index].set_title(
            #         "{}".format(F_Hplus), fontsize=20
            #         )
            # if (col_index == F_Hplus_array.size - 1):
            #     ax[row_index, col_index].set_ylabel(
            #         "{}".format(F_atp), fontsize=20
            #         )

            #     ax[row_index, col_index].yaxis.set_label_position("right")

    for i in range(F_atp_array.size):
        for j in range(F_Hplus_array.size):

            if (i == 0):
                ax[i, j].set_title(
                    "{}".format(F_Hplus_array[j]), fontsize=20
                    )
            if (j == F_Hplus_array.size - 1):
                ax[i, j].set_ylabel(
                    "{}".format(F_atp_array[i]), fontsize=20
                    )
                ax[i, j].yaxis.set_label_position("right")

            ax[i, j].grid(True)
            ax[i, j].tick_params(labelsize=15)
            ax[i, j].yaxis.offsetText.set_fontsize(12)
            ax[i, j].ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    fig.text(
        0.03, 0.54,
        r"$-\left(\mathcal{P}_{\mathrm{out}}/\mathcal{P}_{\mathrm{in}}\right)$",
        fontsize=20, rotation="vertical"
        )
    fig.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=20, rotation="vertical"
        )
    fig.text(0.49, 0.03, r"$E_{\mathrm{couple}}$", fontsize=20)
    fig.text(0.51, 0.95, r"$F_{H^{+}}$", fontsize=20)

    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(target_dir + "/efficiency_scan_figure.pdf")

if __name__ == "__main__":
    target_dir = "/Users/jlucero/data_to_not_upload/2019-03-12/"
    # calculate_flux_power_and_efficiency(target_dir)
    # plot_energy()
    # plot_probability()
    # plot_efficiency(target_dir)
    # plot_flux_scan(target_dir)
    # plot_power_scan(target_dir)
    plot_efficiency_scan(target_dir)
