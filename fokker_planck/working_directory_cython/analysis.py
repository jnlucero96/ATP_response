#!/anaconda3/bin/python
from math import cos, sin, pi
from numpy import (
    array, linspace, arange, loadtxt, vectorize, pi as npi, exp,
    empty, log, log2, finfo, true_divide, asarray, where
    )
from scipy.integrate import trapz
from matplotlib import rcParams, rc, ticker, colors, cm
from matplotlib.style import use
from matplotlib.pyplot import subplots, close
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import getcwd
from datetime import datetime

from utilities import calc_flux, calc_learning_rate

use('seaborn-paper')
rc('text', usetex=True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath}", r"\usepackage{lmodern}",
    r"\usepackage{siunitx}", r"\usepackage{units}",
    r"\usepackage{physics}", r"\usepackage{bm}"
]

# declare global arrays
Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
# F_Hplus_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
# F_atp_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
F_Hplus_array = array([0.0, 2.0, 4.0, 8.0])
F_atp_array = array([-8.0, -4.0, -2.0, 0.0])[::-1]

def set_params():

    N = 360
    E0 = 4.0
    Ecouple = 128.0
    E1 = 4.0
    F_Hplus = 8.0
    F_atp = -2.0
    num_minima = 3.0
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

def plot_energy(target_dir):

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
        target_dir + "/energy_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_E_Hplus_{3}_E_atp_{4}_minima_{5}_phase_{6}".format(
            E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
            ) + "_figure.pdf"
        )

def plot_probability_against_reference(ref_dir, target_dir):

    [
        N, E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = 2*pi/N

    ref_file_name = (
        "/reference_"
        + "E0_{0}_F_Hplus_{1}_F_atp_{2}_minima_{3}_outfile.dat".format(
            E0, F_Hplus, F_atp, num_minima
            )
        )

    input_file_name = (
        "/reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}".format(
            E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
        ) + "_outfile.dat"
        )

    prob_array_2D = loadtxt(
        target_dir + input_file_name,
        usecols=(0,)
        ).reshape((N, N));

    marginal_prob_X = trapz(prob_array_2D, axis=1, dx=dx)
    marginal_prob_Y = trapz(prob_array_2D, axis=0, dx=dx)

    prob_array_1D = loadtxt(
        ref_dir + ref_file_name,
        usecols=(0,)
    )

    positions = linspace(0.0, 2*pi-dx, N)

    fig, ax = subplots(1, 1, figsize=(10,10))

    ax.semilogy(
        positions*(180/npi), marginal_prob_X, lw=3.0, color='black', ls='-',
        label=r"$\int\dd{y}\ P_{2\mathrm{D}}(x,y)$"
        )
    ax.semilogy(
        positions*(180/npi), marginal_prob_Y, lw=3.0, color='darkgoldenrod', ls='dotted',
        label=r"$\int\dd{x}\ P_{2\mathrm{D}}(x,y)$"
        )
    ax.semilogy(
        positions*(180/npi), prob_array_1D, lw=3.0, color='red', ls='dashed',
        label=r"$P_{1\mathrm{D}}(x,y)$"
        )

    ax.set_xticks(arange(0, 361, 60))
    ax.set_xlim([0.0,360.0])
    ax.set_ylabel(r"$P(x)$", fontsize=20)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.tick_params(labelsize=16, axis='both')
    ax.grid(True)
    ax.legend(loc=0, prop={"size":12})

    fig.tight_layout()
    fig.savefig(
        target_dir + "/probability_comparison_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_E_Hplus_{3}_E_atp_{4}".format(
            E0, Ecouple, E1, F_Hplus, F_atp
            ) + "_sfigure.pdf"
        )

def plot_probability(target_dir):

    [
        N, E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = 2*pi/N

    input_file_name = (
        "/reference_"
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
        target_dir + "/probability_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_E_Hplus_{3}_E_atp_{4}".format(
            E0, Ecouple, E1, F_Hplus, F_atp
            ) + "_sfigure.pdf"
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
        "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}".format(
            E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
            ) + "_figure.pdf"
        )

def plot_efficiency_against_ratio(target_dir):

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        __, __, __, __
        ] = set_params()

    dx = (2*pi)/N

    x_array = empty(F_Hplus_array.size*F_atp_array.size)
    efficiency = empty(F_Hplus_array.size*F_atp_array.size)

    for ii, F_Hplus in enumerate(F_Hplus_array):
        for jj, F_atp in enumerate(F_atp_array):

            index = ii*F_Hplus_array.size + jj

            Ecouple_array, efficiency_array = loadtxt(
                target_dir + input_file_name.format(
                    E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                    ),
                unpack=True, usecols=(0,5)
            )

            if (abs(F_Hplus) < abs(F_atp)):
                x_array[index] = (F_Hplus/F_atp)**2
                efficiency[index] = efficiency_array[0]
            elif (abs(F_Hplus) > abs(F_atp)):
                x_array[index] = (F_atp/F_Hplus)**2
                efficiency[index] = efficiency_array[0]
            else:
                continue

    # print(x_array); print(efficiency); exit(0)

    fig, ax = subplots(1, 1, figsize=(10,10))

    l0 = ax.scatter(
        x_array, efficiency
        )

    ax.set_ylabel(
        r"$-\left(\mathcal{P}_{\mathrm{H}^{+}}/\mathcal{P}_{\mathrm{atp}}\right)$",
        fontsize=20
        )
    ax.set_xlabel(
        r"$\left(\mathrm{F}_{\mathrm{H}^{+}}/\mathrm{F}_{\mathrm{atp}}\right)^{2}$",
        fontsize=20
        )
    ax.tick_params(labelsize=16, axis='both')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        target_dir + "/efficiency_against_ratio_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_minima_{3}_phase_{4}".format(
            E0, Ecouple, E1, num_minima, phase_shift
            ) + "_figure.pdf"
        )

def plot_lr(target_dir):

    [
        N, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    dx = (2*pi)/N

    positions = linspace(0.0, 2*pi-dx, N)

    learning_rates = empty((F_atp_array.size,F_Hplus_array.size))

    for ii, F_atp in enumerate(F_atp_array):
        for jj, F_Hplus in enumerate(F_Hplus_array):

            reference_file_name = (
                "reference_"
                + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}".format(
                    E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                ) + "_outfile.dat")
            prob_ss_array, __, __, force1_array, force2_array = loadtxt(
                target_dir + reference_file_name.format(
                    E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                    ), unpack=True
            )

            prob_ss_array = prob_ss_array.reshape((N,N))
            force1_array = force1_array.reshape((N,N))
            force2_array = force2_array.reshape((N,N))

            flux_array = empty((2,N,N))
            calc_flux(
                positions, prob_ss_array, force1_array, force2_array,
                flux_array, m1, m2, gamma, beta, N, dx, 0.001
                )

            Ly = empty((N,N))
            calc_learning_rate(
                prob_ss_array, prob_ss_array.sum(axis=1),
                flux_array[1,:,:],
                Ly,
                N, dx
               )

            learning_rates[ii, jj] = trapz(trapz(Ly, axis=1, dx=dx), dx=dx)

    vmin=learning_rates.min()
    vmax=learning_rates.max()

    limit=max(abs(vmin), abs(vmax))

    # prepare figure
    fig, ax = subplots(1, 1, figsize=(10,10))
    im = ax.imshow(
        learning_rates.T, cmap=cm.get_cmap('coolwarm'),
        extent=(
            F_atp_array[0], F_atp_array[-1],
            F_Hplus_array[0], F_Hplus_array[-1]
            ),
        vmin=-limit, vmax=limit
        )
    ax.tick_params(labelsize=20)
    ax.set_ylabel(r"$\mathrm{F}_{\mathrm{H}^{+}}$", fontsize=28)
    ax.set_xlabel(r"$\mathrm{F}_{\mathrm{atp}}$", fontsize=28)


    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, ax=ax, cax=cax)
    fig.savefig(
        target_dir
        + "learning_rate_E0_{0}_Ecouple_{1}_E1_{2}_minima_{3}_phase_{4}".format(
                E0, Ecouple, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

# ============================================================================
# ==============
# ============== SCAN PLOTS
# ==============
# ============================================================================

def plot_probability_scan(target_dir):

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

    fig, ax = subplots(
        F_Hplus_array.size, F_atp_array.size,
        figsize=(15,15), sharex='col', sharey='all'
        )

    positions = linspace(0.0, 2*pi-dx, N)

    to_plot = []
    to_plot_eq = []

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

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

            to_plot.append(prob_array)
            to_plot_eq.append(prob_eq_array)

    vmin_ss = min([array_to_plot.min() for array_to_plot in to_plot])
    vmax_ss = max([array_to_plot.max() for array_to_plot in to_plot])
    vmin_eq = min([array_to_plot_eq.min() for array_to_plot_eq in to_plot_eq])
    vmax_eq = max([array_to_plot_eq.max() for array_to_plot_eq in to_plot_eq])

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

            cl0 = ax[row_index, col_index].contourf(
                positions, positions,
                (to_plot_eq[row_index*F_atp_array.size + col_index]).T, 30,
                vmin=vmin_eq, vmax=vmax_eq, cmap=cm.get_cmap("gnuplot")
                )
            ax[row_index, col_index].contour(
                positions, positions,
                (to_plot[row_index*F_atp_array.size + col_index]).T, 15,
                vmin=vmin_ss, vmax=vmax_ss, cmap=cm.get_cmap("cool"),
                linestyles='dashed'
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
            ax[i, j].ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    fig.text(
        0.03, 0.51, r"$x_{2}$", fontsize=28, rotation="vertical"
        )
    fig.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=28, rotation="vertical"
        )
    fig.text(0.505, 0.03, r"$x_{1}$", fontsize=28)
    fig.text(0.50, 0.97, r"$F_{H^{+}}$", fontsize=28)

    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(
        target_dir
        + "/probability_scan_E0_{0}_E1_{1}_Ecouple_{2}_minima_{3}_phase_{4}_figure.pdf".format(
                E0, E1, Ecouple, num_minima, phase_shift
            )
        )

def plot_flux_scan(target_dir):

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    dx = (2*pi)/N

    fig, ax = subplots(
        F_Hplus_array.size, F_atp_array.size,
        figsize=(15,15), sharex='col', sharey='all'
        )
    fig2, ax2 = subplots(
        F_Hplus_array.size, F_atp_array.size,
        figsize=(15,15), sharex='col', sharey='all'
        )

    positions = linspace(0.0, 2*pi-dx, N)

    flux1=[]
    flux2=[]

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

            input_file_name = (
                "reference_"
                + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}".format(
                    E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                ) + "_outfile.dat")

            prob_ss_array, __, __, force1_array, force2_array = loadtxt(
                target_dir + input_file_name.format(
                    E0,Ecouple,E1,F_Hplus,F_atp,num_minima,phase_shift
                    ), unpack=True
            )

            prob_ss_array = prob_ss_array.reshape((N,N))
            force1_array = force1_array.reshape((N,N))
            force2_array = force2_array.reshape((N,N))

            flux_array = empty((2,N,N))
            calc_flux(
                positions, prob_ss_array, force1_array, force2_array,
                flux_array, m1, m2, gamma, beta, N, dx, 0.001
                )

            flux_array = asarray(flux_array)
            flux1.append(flux_array[0,...])
            flux2.append(flux_array[1,...])

    flux1_min = min([flux1_array.min() for flux1_array in flux1])
    flux1_max = max([flux1_array.max() for flux1_array in flux1])
    flux2_min = min([flux2_array.min() for flux2_array in flux2])
    flux2_max = max([flux2_array.max() for flux2_array in flux2])

    limit = max(abs(flux1_min), abs(flux1_max), abs(flux2_min), abs(flux2_max))

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

            ax[row_index, col_index].contourf(
                positions, positions, (flux1[
                    row_index*F_atp_array.size + col_index
                    ]).T,
                vmin=-limit, vmax=limit,
                cmap=cm.get_cmap('seismic')
                )

            ax2[row_index, col_index].contourf(
                positions, positions, (flux2[
                    row_index*F_atp_array.size + col_index
                    ]).T,
                vmin=-limit, vmax=limit,
                cmap=cm.get_cmap('seismic')
                )

            if (row_index == 0):
                ax[row_index, col_index].set_title(
                    "{}".format(F_Hplus), fontsize=20
                    )

                ax2[row_index, col_index].set_title(
                    "{}".format(F_Hplus), fontsize=20
                    )

            if (col_index == F_Hplus_array.size - 1):
                ax[row_index, col_index].set_ylabel(
                    "{}".format(F_atp), fontsize=20
                    )
                ax[row_index, col_index].yaxis.set_label_position("right")

                ax2[row_index, col_index].set_ylabel(
                    "{}".format(F_atp), fontsize=20
                    )
                ax2[row_index, col_index].yaxis.set_label_position("right")

    for i in range(F_atp_array.size):
        for j in range(F_Hplus_array.size):
            ax[i, j].grid(True)
            ax[i, j].tick_params(labelsize=15)
            ax[i, j].yaxis.offsetText.set_fontsize(12)
            ax[i, j].ticklabel_format(style="sci", axis="y", scilimits=(0,0))

            ax2[i, j].grid(True)
            ax2[i, j].tick_params(labelsize=15)
            ax2[i, j].yaxis.offsetText.set_fontsize(12)
            ax2[i, j].ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    fig.text(
        0.03, 0.51, r"$x_{2}$", fontsize=28, rotation="vertical"
        )
    fig.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=28, rotation="vertical"
        )
    fig.text(0.49, 0.03, r"$x_{1}$", fontsize=28)
    fig.text(0.50, 0.97, r"$F_{H^{+}}$", fontsize=28)

    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(
        target_dir
        + "/flux1_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}_figure.pdf".format(
                E0, E1, num_minima, phase_shift
            )
        )

    fig2.text(
        0.03, 0.51, r"$\langle J\rangle$", fontsize=28, rotation="vertical"
        )
    fig2.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=28, rotation="vertical"
        )
    fig2.text(0.49, 0.03, r"$E_{\mathrm{couple}}$", fontsize=28)
    fig2.text(0.50, 0.97, r"$F_{H^{+}}$", fontsize=28)

    fig2.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig2.savefig(
        target_dir
        + "/flux2_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}_figure.pdf".format(
                E0, E1, num_minima, phase_shift
            )
        )

def plot_integrated_flux_scan(target_dir):

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
        0.03, 0.51, r"$\langle J\rangle$", fontsize=28, rotation="vertical"
        )
    fig.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=28, rotation="vertical"
        )
    fig.text(0.49, 0.03, r"$E_{\mathrm{couple}}$", fontsize=28)
    fig.text(0.50, 0.97, r"$F_{H^{+}}$", fontsize=28)

    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(
        target_dir
        + "/integrated_flux_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}_figure.pdf".format(
                E0, E1, num_minima, phase_shift
            )
        )

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
        0.03, 0.51, r"$\mathcal{P}$", fontsize=28, rotation="vertical"
        )
    fig.text(
        0.96, 0.51, r"$F_{\mathrm{atp}}$", fontsize=28, rotation="vertical"
        )
    fig.text(0.49, 0.03, r"$E_{\mathrm{couple}}$", fontsize=28)
    fig.text(0.50, 0.97, r"$F_{H^{+}}$", fontsize=28)

    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(
        target_dir
        + "/power_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}_figure.pdf".format(
                E0, E1, num_minima, phase_shift
            )
        )

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
            ax[i, j].yaxis.offsetText.set_fontsize(15)
            ax[i, j].ticklabel_format(style="sci", axis="y", scilimits=(0,0))


    fig.tight_layout()

    left=0.1
    right=0.925
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.text(
        0.03, 0.60*(top-bottom),
        r"$-\left(\mathcal{P}_{\mathrm{out}}/\mathcal{P}_{\mathrm{in}}\right)$",
        fontsize=28, rotation="vertical", ha="center", va="center"
        )
    fig.text(
        0.97, 0.625*(top-bottom), r"$F_{\mathrm{atp}}$",
        fontsize=28, rotation="vertical", ha="center", va="center"
        )
    fig.text(
        0.625*(right-left), 0.03, r"$E_{\mathrm{couple}}$",
        fontsize=28, ha="center", va="center"
        )
    fig.text(
        0.625*(right-left), 0.97, r"$F_{H^{+}}$",
        fontsize=28, ha="center", va="center"
        )

    fig.savefig(
        target_dir
        + "/efficiency_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}_figure.pdf".format(
            E0, E1, num_minima, phase_shift
            )
        )


if __name__ == "__main__":
    ref_dir = "/Users/jlucero/data_to_not_upload/2019-03-20/"
    target_dir = "/Users/jlucero/data_to_not_upload/2019-03-14/"
    # calculate_flux_power_and_efficiency(target_dir)
    # plot_energy()
    # plot_probability()
    plot_probability_against_reference(ref_dir, target_dir)
    # plot_efficiency(target_dir)
    # plot_efficiency_against_ratio(target_dir)
    # plot_lr(target_dir)
    # plot_probability_scan(target_dir)
    # plot_flux_scan(target_dir)
    # plot_integrated_flux_scan(target_dir)
    # plot_power_scan(target_dir)
    # plot_efficiency_scan(target_dir)
