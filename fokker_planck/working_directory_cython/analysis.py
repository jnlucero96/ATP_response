#!/anaconda3/bin/python
from math import cos, sin, pi, sqrt
from numpy import (
    array, linspace, arange, loadtxt, vectorize, pi as npi, exp, zeros,
    empty, log, log2, finfo, true_divide, asarray, where, partition, isnan, nan,
    ones, argmax, argmin, set_printoptions, diagonal
    )
from numpy.random import random
from scipy.integrate import trapz
from scipy.signal import correlate2d
from matplotlib import rcParams, rc, ticker, colors, cm
from matplotlib.style import use
from matplotlib.pyplot import subplots, close
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import getcwd
from datetime import datetime

from utilities import (
    calc_flux, calc_learning_rate, calc_derivative_pxgy, step_probability_X
    )

use('seaborn-paper')
rc('text', usetex=True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath}", r"\usepackage{lmodern}",
    r"\usepackage{siunitx}", r"\usepackage{units}",
    r"\usepackage{physics}", r"\usepackage{bm}"
]
set_printoptions(linewidth=512)


# declare global arrays
Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
# F_Hplus_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
# F_atp_array = array([-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
F_Hplus_array = array([0.0, 2.0, 4.0, 8.0])
F_atp_array = array([-8.0, -4.0, -2.0, 0.0])[::-1]

def set_params():

    N = 360
    E0 = 0.0
    Ecouple = 16.0
    E1 = 0.0
    F_Hplus = 4.0
    F_atp = -1.0
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

                    print(f"Calculating flux for F_atp = {F_atp}, F_Hplus = {F_Hplus}, Ecouple = {Ecouple}")

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
                        flux_array, m1, m2, gamma, beta, N, dx
                        )

                    flux_array = asarray(flux_array)/(dx*dx)
                    force1_array = asarray(force1_array)
                    force2_array = asarray(force2_array)

                    integrate_flux_X[ii] = (1./(2*pi))*trapz(
                        trapz(flux_array[0,...], dx=dx, axis=1), dx=dx
                        )
                    integrate_flux_Y[ii] = (1./(2*pi))*trapz(
                        trapz(flux_array[1,...], dx=dx, axis=0), dx=dx
                        )
                    # integrate_flux_X[ii] = flux_array[0,...].sum(axis=None)/(N*N)
                    # integrate_flux_Y[ii] = flux_array[1,...].sum(axis=None)/(N*N)
                    integrate_power_X[ii] = integrate_flux_X[ii]*F_Hplus
                    integrate_power_Y[ii] = integrate_flux_Y[ii]*F_atp

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
    # ax.plot(
    #     positions_deg, V[0,:], lw=3.0,
    #     )
    #ax.set_ylabel(r"$x_{2}$", fontsize=20)
    # ax.set_ylabel(r"$V_{\mathrm{internal}}(x_{i})$", fontsize=20)
    # ax.set_xlabel(r"$x_{i}$", fontsize=20)
    ax.tick_params(axis="both", labelsize=30)
    ax.set_xticks(arange(0.0, 361, 60))
    ax.set_yticks(arange(0.0, 361, 60))
    # ax.set_yticks([0.0, E0])
    # ax.set_yticklabels([r"$0.0$", r"$A$"])
    ax.set_xlim([0.0, 360.0])
    # ax.set_ylim([0.0, E0+0.1])
    ax.set_ylim([0.0, 360.0])

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

    cbar=fig.colorbar(ct, ax=ax)
    cbar.ax.tick_params(labelsize=20)
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
            2.0*E0, F_Hplus, F_atp, num_minima
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
        positions*(180/npi), prob_array_2D.sum(axis=1), lw=3.0, color='black', ls='-',
        label=r"$\int\dd{y}\ P_{2\mathrm{D}}(x,y)$"
        )
    ax.semilogy(
        positions*(180/npi), prob_array_2D.sum(axis=0), lw=3.0, color='darkgoldenrod', ls='dotted',
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
        "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}".format(
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
    # to_plot = [
    #     prob_array/prob_array.sum(axis=0)
    # ]
    vmin = min([array_to_plot.min() for array_to_plot in to_plot])
    vmax = max([array_to_plot.max() for array_to_plot in to_plot])

    positions = linspace(0.0, 2*pi-dx, N)
    positions_deg = positions * (180.0/npi)

    fig, ax = subplots(1, 1, figsize=(10,10))

    cl0 = ax.contourf(
        positions, positions, prob_eq_array.T, 30,
        vmin=vmin, vmax=vmax, cmap=cm.get_cmap("gnuplot")
        )
    pxgy = (prob_array/(prob_array.sum(axis=0)))
    print(pxgy.sum(axis=0))

    cl0=ax.contourf(
        positions_deg, positions_deg, prob_eq_array.T, 30,
        vmin=vmin, vmax=vmax, cmap=cm.get_cmap("gnuplot")
        )
    cl0=ax.contourf(
        positions_deg, positions_deg, prob_array.T, 30,
        vmin=vmin, vmax=vmax, cmap=cm.get_cmap("gnuplot")
        )

    ax.set_xticks(arange(0, 361, 60))
    ax.set_yticks(arange(0, 361, 60))
    # ax.set_ylabel(r"$\theta_{1}$", fontsize=20)
    # ax.set_xlabel(r"$\theta_{0}$", fontsize=20)
    ax.tick_params(labelsize=30, axis='both')
    ax.grid(True)

    sfmt=ticker.ScalarFormatter()
    sfmt.set_powerlimits((0, 0))
    cbar=fig.colorbar(cl0, ax=ax, format=sfmt)
    cbar.ax.tick_params(labelsize=24, axis='y')
    cbar.ax.yaxis.offsetText.set_fontsize(16)

    fig.tight_layout()
    fig.savefig(
        target_dir + "/probability_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_E_Hplus_{3}_E_atp_{4}".format(
            E0, Ecouple, E1, F_Hplus, F_atp
            ) + "_sfigure.pdf"
        )

def plot_power(target_dir):

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

    Ecouple_array, integrate_power_X, integrate_power_Y = loadtxt(
        target_dir + input_file_name.format(
            E0, E1, F_Hplus, F_atp, num_minima, phase_shift
            ),
        unpack=True, usecols=(0,3,4)
    )

    fig, ax = subplots(1, 1, figsize=(10,10))

    ax.axhline(0.0, ls='--', lw=3.0)
    ax.plot(
        Ecouple_array, integrate_power_X, lw=3.0, color='black',
        label=r'$\mathcal{P}_{o}$'
        )
    ax.plot(
        Ecouple_array, integrate_power_Y, lw=3.0, color='red',
        label=r'$\mathcal{P}_{1}$'
        )

    # ax.set_ylabel(
    #     r"$\mathcal{P}_{\mathrm{atp}}\right)$",
    #     fontsize=20
    #     )
    # ax.set_xlabel(r"$E_{\mathrm{couple}}$", fontsize=20)
    ax.tick_params(labelsize=30, axis='both')
    # ax.grid(True)
    ax.legend(loc=0, prop={'size':28})

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    ax.yaxis.offsetText.set_fontsize(22)


    fig.tight_layout()
    fig.savefig(
        target_dir + "/power_" +
        "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}".format(
            E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
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
        Ecouple_array, efficiency_ratio, lw=3.0, color='black'
        )

    # ax.set_ylabel(
    #     r"$-\left(\mathcal{P}_{\mathrm{H}^{+}}/\mathcal{P}_{\mathrm{atp}}\right)$",
    #     fontsize=20
    #     )
    # ax.set_xlabel(r"$E_{\mathrm{couple}}$", fontsize=20)
    ax.tick_params(labelsize=30, axis='both')
    # ax.grid(True)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    ax.yaxis.offsetText.set_fontsize(22)

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

def plot_flux(target_dir):

    [
        __, E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    reference_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )

    prob_ss_array, __, __, force1_array, force2_array = loadtxt(
        target_dir + reference_file_name.format(
            E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
            ), unpack=True
    )

    N = int(sqrt(prob_ss_array.size))
    dx = (2*pi)/N
    positions = linspace(0.0, 2*pi-dx, N)
    positions_deg = positions*(180.0/npi)

    prob_ss_array = prob_ss_array.reshape((N, N))
    force1_array = force1_array.reshape((N, N))
    force2_array = force2_array.reshape((N, N))

    flux_array = empty((2, N, N))
    # flux_array = ones((2,N,N))
    calc_flux(
        positions, prob_ss_array, force1_array, force2_array,
        flux_array, m1, m2, gamma, beta, N, dx
        )

    limit=flux_array.__abs__().max()

    # prepare figure
    fig, ax = subplots(1, 2, figsize=(20,10), sharey='row')
    ax[0].contourf(
        positions_deg, positions_deg, flux_array[0, ...].T, 30,
        vmin=-limit, vmax=limit, cmap=cm.get_cmap("summer")
        )
    ax[0].set_title(r"$J_{1}(\vb{x})$", fontsize=32)
    im = ax[1].contourf(
        positions_deg, positions_deg, flux_array[1, ...].T, 30,
        vmin=-limit, vmax=limit, cmap=cm.get_cmap("summer")
        )
    ax[1].set_title(r"$J_{2}(\vb{x})$", fontsize=32)
    
    for axis in ax:
        axis.set_yticks(arange(0, 361, 60))
        axis.set_xticks(arange(0, 361, 60))
        axis.tick_params(labelsize=30)

    cax = fig.add_axes([0.90, 0.12, 0.01, 0.8])
    cbar1 = fig.colorbar(
        im, cax=cax, orientation='vertical',
        ax=ax
    )
    cbar1.set_label(
        r"$J_{i}(\vb{x})$",
        fontsize=32
        )
    # cbar1.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar1.formatter.set_scientific(True)
    cbar1.formatter.set_powerlimits((0,0))
    cbar1.ax.tick_params(labelsize=24)
    cbar1.ax.yaxis.offsetText.set_fontsize(24)
    cbar1.ax.yaxis.offsetText.set_x(5.0)
    cbar1.update_ticks()
    fig.tight_layout()

    # y-axis label
    fig.text(
        0.025, 0.51,
        r'$x_{2}\ (\mathrm{units\ of\ }\mathrm{deg})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig.text(
        0.48, 0.03,
        r'$x_{1} (\mathrm{units\ of\ }\mathrm{deg})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.1  # the left side of the subplots of the figure
    right = 0.89    # the right side of the subplots of the figure
    bottom = 0.12 # the bottom of the subplots of the figure
    top = 0.92     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.savefig(
        target_dir
        + f"flux_E0_{E0}_Ecouple_{Ecouple}_E1_{E1}_F_Hplus_{F_Hplus}_F_atp_{F_atp}_minima_{num_minima}_phase_{phase_shift}"
        + "_figure.pdf"
        )


def plot_lr(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    reference_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )

    learning_rates = empty((F_atp_array.size,F_Hplus_array.size))

    for ii, F_atp in enumerate(F_atp_array):
        for jj, F_Hplus in enumerate(F_Hplus_array):

            prob_ss_array, __, __, force1_array, force2_array = loadtxt(
                target_dir + reference_file_name.format(
                    E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                    ), unpack=True
            )

            if (ii==0 and jj==0):
                N = int(sqrt(prob_ss_array.size))
                dx = (2*pi)/N
                positions = linspace(0.0, 2*pi-dx, N)

            prob_ss_array = prob_ss_array.reshape((N,N))
            force1_array = force1_array.reshape((N,N))
            force2_array = force2_array.reshape((N,N))

            flux_array = empty((2,N,N))
            calc_flux(
                positions, prob_ss_array, force1_array, force2_array,
                flux_array, m1, m2, gamma, beta, N, dx
                )

            Ly = empty((N,N))
            calc_learning_rate(
                prob_ss_array, prob_ss_array.sum(axis=0),
                flux_array[1,:,:],
                Ly,
                N, dx
               )

            learning_rates[ii, jj] = Ly.sum(axis=None)

    vmin=learning_rates.min()
    vmax=learning_rates.max()

    limit=learning_rates.__abs__().max()

    # prepare figure
    fig, ax = subplots(1, 1, figsize=(10,10))
    im = ax.imshow(
        learning_rates.T,
        vmin=-limit, vmax=limit,
        cmap=cm.get_cmap("coolwarm")
        )
    ax.set_yticks(list(range(F_Hplus_array.size)))
    ax.set_yticklabels(F_Hplus_array)
    ax.set_xticks(list(range(F_atp_array.size)))
    ax.set_xticklabels(F_atp_array)
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

def plot_energy_scan(target_dir):

    [
        N, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    dx = (2*pi)/N
    positions = linspace(0.0, 2*pi-dx, N)
    positions_deg = positions * (180.0/npi)

    vec_landscape = vectorize(landscape)

    energies = zeros((Ecouple_array.size, N, N))

    for ee, Ecouple in enumerate(Ecouple_array):

        energies[ee, ...] = vec_landscape(
            E0, Ecouple, E1, num_minima, phase_shift,
            positions[:,None], positions[None,:]
            )

    limit=energies.__abs__().max()

    # prepare figure
    fig, ax = subplots(2, 4, figsize=(15,10), sharey='all')
    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        cs = ax[xloc, yloc].contourf(
            positions_deg, positions_deg,
            energies[ee,...].T,
            vmin=0.0, vmax=limit,
            cmap=cm.get_cmap("afmhot")
            )

        ax[xloc, yloc].set_xticks(range(0,360,120))
        ax[xloc, yloc].set_yticks(range(0,360,120))
        ax[xloc, yloc].tick_params(labelsize=20)
        ax[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=32
            )

    cax = fig.add_axes([0.89, 0.10, 0.01, 0.85])
    cbar = fig.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax
    )
    cbar.set_label(r'$\beta V_{\mathrm{tot}}\left(\vb{x}\right)$', fontsize=26)
    # cbar.formatter.set_scientific(True)
    # cbar.formatter.set_powerlimits((0,0))
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.yaxis.offsetText.set_fontsize(22)
    cbar.update_ticks()
    fig.tight_layout()

    # y-axis label
    fig.text(
        0.025, 0.51,
        r'$x_{2}\ (\mathrm{units\ of\ }\mathrm{deg})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig.text(
        0.47, 0.03,
        r'$x_{1}\ (\mathrm{units\ of\ }\mathrm{deg})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.075  # the left side of the subplots of the figure
    right = 0.87    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.95     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.savefig(
        target_dir
        + "/energy_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_probability_eq_scan(target_dir):

    [
        N, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    reference_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )

    dx = (2*pi)/N
    positions = linspace(0.0, 2*pi-dx, N)
    positions_deg = positions * (180.0/npi)

    equilibrium_probabilities = zeros((Ecouple_array.size, N, N))

    for ee, Ecouple in enumerate(Ecouple_array):

        prob_eq_array = loadtxt(
            target_dir + reference_file_name.format(
                E0, Ecouple, E1, 0.0, 0.0, num_minima, phase_shift
                ), usecols=(1,)
        )

        equilibrium_probabilities[ee, ...] = prob_eq_array.reshape((N,N))

    limit=equilibrium_probabilities.__abs__().max()

    # prepare figure
    fig, ax = subplots(2, 4, figsize=(15,10), sharey='all')
    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        cs = ax[xloc, yloc].contourf(
            positions_deg, positions_deg,
            equilibrium_probabilities[ee,...].T, 30,
            vmin=0.0, vmax=limit,
            cmap=cm.get_cmap("Purples_r")
            )

        ax[xloc, yloc].set_xticks(range(0,360,120))
        ax[xloc, yloc].set_yticks(range(0,360,120))
        ax[xloc, yloc].tick_params(labelsize=20)
        ax[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=28
            )

    cax = fig.add_axes([0.92, 0.10, 0.01, 0.85])
    cbar = fig.colorbar(
        cs, cax=cax, orientation='vertical',
        ax=ax
    )
    tick_array = [0.0, 0.00016, 0.00032, 0.00048, 0.00064, 0.00080]
    cbar.set_ticks(tick_array)
    cbar.set_label(r'$\pi\left(\vb{x}\right)$', fontsize=32)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0,0))
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.yaxis.offsetText.set_fontsize(24)
    cbar.ax.yaxis.offsetText.set_x(5.0)
    cbar.update_ticks()

    fig.tight_layout()

    # y-axis label
    fig.text(
        0.02, 0.51,
        r'$x_{2}\ (\mathrm{units\ of\ }\mathrm{deg})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig.text(
        0.49, 0.03,
        r'$x_{1}\ (\mathrm{units\ of\ }\mathrm{deg})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.07  # the left side of the subplots of the figure
    right = 0.90    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.95     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.savefig(
        target_dir
        + "/probability_eq_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

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
        F_atp_array.size, F_Hplus_array.size,
        figsize=(15,15), sharex='col', sharey='all'
        )

    positions = linspace(0.0, 2*pi-dx, N) * (180/npi)

    to_plot = []

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

    abs_array_maxes = [array_to_plot.__abs__().max() for array_to_plot in to_plot]
    vmax_ss = max(abs_array_maxes)
    # find the argmax of the probability arrays
    f = lambda i: abs_array_maxes[i] 
    vmax_ss_loc = max(range(len(abs_array_maxes)), key=f)

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

            loc = row_index*F_atp_array.size + col_index
            if (loc == vmax_ss_loc): 
                cs=ax[row_index, col_index].contourf(
                    positions, positions,
                    (to_plot[loc]).T, 30,
                    vmin=0.0, vmax=vmax_ss, cmap=cm.get_cmap("gnuplot")
                    )
            else:
                ax[row_index, col_index].contourf(
                    positions, positions,
                    (to_plot[loc]).T, 30,
                    vmin=0.0, vmax=vmax_ss, cmap=cm.get_cmap("gnuplot")
                    )

            if (row_index == 0):
                ax[row_index, col_index].set_title(
                    "{}".format(int(F_Hplus)), fontsize=28
                    )
            if (col_index == F_Hplus_array.size - 1):
                ax[row_index, col_index].set_ylabel(
                    "{}".format(int(F_atp)), fontsize=28
                    )
                ax[row_index, col_index].yaxis.set_label_position("right")

    for i in range(F_atp_array.size):
        for j in range(F_Hplus_array.size):

            ax[i, j].tick_params(labelsize=22)
            ax[i, j].set_xticks(range(0,360,120))
            ax[i, j].set_yticks(range(0,360,120))

    cax = fig.add_axes([0.88, 0.10, 0.02, 0.825])
    cbar = fig.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax
    )
    # ticks = [
    #     0.0, 0.5e-04, 1.0e-04, 1.5e-04
    #     ]
    # cbar.set_ticks(ticks)
    cbar.set_label(r'$\rho^{\mathrm{SS}}(\vb{x})$', fontsize=32)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0,0))
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.yaxis.offsetText.set_fontsize(24)
    cbar.ax.yaxis.offsetText.set_x(2.75)
    cbar.update_ticks()

    fig.text(
        0.02, 0.51, r"$x_{2}\ (\mathrm{units\ of\ }\mathrm{deg})$", fontsize=36, rotation="vertical",
        va='center', ha='center'
        )
    fig.text(
        0.855, 0.51, r"$\beta\psi_{2}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$", fontsize=36, rotation="vertical",
        va='center', ha='center'
        )
    fig.text(
        0.455, 0.03, r"$x_{1}\ (\mathrm{units\ of\ }\mathrm{deg})$", fontsize=36, va='center', ha='center'
        )
    fig.text(
        0.45, 0.97, r"$\beta\psi_{1}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$", fontsize=36, va='center', ha='center'
        )

    fig.tight_layout()

    left=0.1
    right=0.80
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
    temp_fig, temp_ax = subplots(1,1)

    positions = linspace(0.0, 2*pi-dx, N)

    flux1=empty((F_atp_array.size*F_Hplus_array.size,N,N))
    flux2=empty((F_atp_array.size*F_Hplus_array.size,N,N))

    flux1_maxloc = 0; flux2_maxloc = 0; f1_curr_max = 0.0; f2_curr_max = 0.0

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

            curr_loc = row_index*F_atp_array.size + col_index

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
                flux_array, m1, m2, gamma, beta, N, dx
                )

            # zero out any values that are beyond machine precision
            flux_array[flux_array.__abs__() <= finfo('float64').eps] = 0.0

            flux_array = asarray(flux_array)/(dx*dx)

            if (flux_array[0,...].__abs__().max() > f1_curr_max): 
                f1_curr_max = flux_array[0,...].__abs__().max()
                flux1_maxloc = curr_loc
            if (flux_array[1,...].__abs__().max() > f2_curr_max): 
                f1_curr_max = flux_array[1,...].__abs__().max()
                flux2_maxloc = curr_loc

            flux1[curr_loc,...] = flux_array[0,...]
            flux2[curr_loc,...] = flux_array[1,...]

    limit = max(flux1.__abs__().max(), flux2.__abs__().max())

    cs = temp_ax.contourf(
        positions, positions, linspace((-1.05)*limit,(1.05)*limit,N*N).reshape(N,N), 
        30, vmin=-limit, vmax=limit, cmap=cm.get_cmap("coolwarm")
        )

    for row_index, F_atp in enumerate(F_atp_array):
        for col_index, F_Hplus in enumerate(F_Hplus_array):

            loc = row_index*F_atp_array.size+col_index

            ax[row_index, col_index].contourf(
                positions, positions, flux1[loc, ...].T, 30,
                vmin=-limit, vmax=limit,
                cmap=cm.get_cmap("coolwarm")
                )

            ax2[row_index, col_index].contourf(
                positions, positions, flux2[loc, ...].T, 30,
                vmin=-limit, vmax=limit,
                cmap=cm.get_cmap("coolwarm")
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

    cax1 = fig.add_axes([0.88, 0.10, 0.02, 0.825])
    cbar1 = fig.colorbar(
        cs, cax=cax1, orientation='vertical', ax=ax2
    )
    cbar1.set_label(r'$J_{1}(\vb{x})$', fontsize=32)
    cbar1.formatter.set_scientific(True)
    cbar1.formatter.set_powerlimits((0,0))
    cbar1.ax.tick_params(labelsize=24)
    cbar1.ax.yaxis.offsetText.set_fontsize(24)
    cbar1.ax.yaxis.offsetText.set_x(2.75)
    cbar1.update_ticks()

    fig.text(
        0.03, 0.51, r"$x_{2}$", fontsize=28, rotation="vertical"
        )
    fig.text(
        0.86, 0.51, r"$\beta\psi_{2}$", fontsize=28, rotation="vertical"
        )
    fig.text(0.49, 0.03, r"$x_{1}$", fontsize=28)
    fig.text(0.50, 0.97, r"$\beta\psi_{1}$", fontsize=28)

    fig.tight_layout()

    left=0.1
    right=0.85
    bottom=0.1
    top=0.925
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig.savefig(
        target_dir
        + f"/flux1_scan_E0_{E0}_E1_{E1}_Ecouple_{Ecouple}_minima_{num_minima}_phase_{phase_shift}_figure.pdf"
        )

    cax2 = fig2.add_axes([0.88, 0.10, 0.02, 0.825])
    cbar2 = fig2.colorbar(
        cs, cax=cax2, orientation='vertical', ax=ax2
    )
    cbar2.set_label(r'$J_{2}(\vb{x})$', fontsize=32)
    cbar2.formatter.set_scientific(True)
    cbar2.formatter.set_powerlimits((0,0))
    cbar2.ax.tick_params(labelsize=24)
    cbar2.ax.yaxis.offsetText.set_fontsize(24)
    cbar2.ax.yaxis.offsetText.set_x(2.75)
    cbar2.update_ticks()

    fig2.text(
        0.03, 0.51, r"$x_{2}$", fontsize=28, rotation="vertical"
        )
    fig2.text(
        0.86, 0.51, r"$\beta\psi_{2}$", fontsize=28, rotation="vertical"
        )
    fig2.text(0.49, 0.03, r"$x_{1}$", fontsize=28)
    fig2.text(0.50, 0.97, r"$\beta\psi_{1}$", fontsize=28)

    fig2.tight_layout()

    left=0.1
    right=0.85
    bottom=0.1
    top=0.925
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    fig2.savefig(
        target_dir
        + f"/flux2_scan_E0_{E0}_E1_{E1}_Ecouple_{Ecouple}_minima_{num_minima}_phase_{phase_shift}_figure.pdf"
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

            ax[row_index, col_index].semilogx(
                Ecouple_array[1:], efficiency_ratio[1:], lw=3.0
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

def plot_efficiency_scan_compare(target_dir):

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    [
        N, __, __, __, __, __, num_minima, phase_shift,
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

            Ecouple_array_2, efficiency_ratio_2 = loadtxt(
                target_dir + input_file_name.format(
                    2.0, 2.0, F_Hplus, F_atp, 3.0, phase_shift
                    ),
                unpack=True, usecols=(0,5)
            )
            Ecouple_array_4, efficiency_ratio_4 = loadtxt(
                target_dir + input_file_name.format(
                    4.0, 4.0, F_Hplus, F_atp, 3.0, phase_shift
                    ),
                unpack=True, usecols=(0,5)
            )
            Ecouple_array_4_minima_9, efficiency_ratio_4_minima_9 = loadtxt(
                target_dir + input_file_name.format(
                    4.0, 4.0, F_Hplus, F_atp, 9.0, phase_shift
                    ),
                unpack=True, usecols=(0,5)
            )

            if (abs(F_Hplus) == abs(F_atp)): continue

            ax[row_index, col_index].semilogx(
                Ecouple_array_2[1:], efficiency_ratio_2[1:], lw=3.0, label=r"$E_{0} = E_{1} = 2.0$"
                )
            ax[row_index, col_index].semilogx(
                Ecouple_array_4[1:], efficiency_ratio_4[1:], lw=3.0, label=r"$E_{0} = E_{1} = 4.0$"
                )
            ax[row_index, col_index].semilogx(
                Ecouple_array_4_minima_9[1:], efficiency_ratio_4_minima_9[1:], lw=3.0, label=r"$n=9$"
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
                ax[i, j].legend(loc=0,prop={'size':10})

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
        + f"/efficiency_scan_compare_phase_{phase_shift}_figure.pdf"
        )

def plot_relative_entropy_lr_scan(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    input_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )

    relative_entropies = zeros((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                prob, prob_eq = loadtxt(
                    target_dir + input_file_name.format(
                        E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ),
                    unpack=True, usecols=(0,1)
                )

                relative_entropies[ee, ii, jj] = (prob*log(prob/prob_eq)).sum(axis=None)


    limit=relative_entropies[~(isnan(relative_entropies))].__abs__().max()

    # prepare figure
    fig, ax = subplots(2, 4, figsize=(15,10), sharey="all")
    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        im = ax[xloc, yloc].imshow(
            relative_entropies[ee,...].T,
            vmin=0.0, vmax=limit,
            cmap=cm.get_cmap("Reds")
            )

        ax[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax[xloc, yloc].tick_params(labelsize=22)
        ax[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=32
            )

    cax = fig.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar1 = fig.colorbar(
        im, cax=cax, orientation='vertical',
        ax=ax
    )
    cbar1.set_label(
        r'$\mathcal{D}\left[\rho^{\mathrm{SS}}(\vb{x})||\pi(\vb{x})\right]$',
        fontsize=32
        )
    cbar1.ax.tick_params(labelsize=24)
    fig.tight_layout()

    # y-axis label
    fig.text(
        0.025, 0.51,
        r'$\beta \psi_{1}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig.text(
        0.48, 0.03,
        r'$\beta \psi_{2}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.065  # the left side of the subplots of the figure
    right = 0.89    # the right side of the subplots of the figure
    bottom = 0.03   # the bottom of the subplots of the figure
    top = 0.98     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.savefig(
        target_dir
        + "/relative_entropy_lr_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_efficiencies_lr_scan(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )


    efficiencies = zeros((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                Ecouple_array_out, efficiency_ratio = loadtxt(
                    target_dir + input_file_name.format(
                        E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ),
                    unpack=True, usecols=(0,5)
                )

                if (abs(F_atp)==abs(F_Hplus)): efficiencies[ee, ii, jj] = nan; continue

                loc=where((Ecouple_array_out-Ecouple).__abs__()<=finfo('float32').eps)[0][0]
                efficiencies[ee, ii, jj] = efficiency_ratio[loc]


    limit=efficiencies[~(isnan(efficiencies))].__abs__().max()

    coolwarm_cmap = cm.get_cmap("coolwarm")
    coolwarm_cmap.set_bad(color='black')

    # prepare figure
    fig, ax = subplots(2, 4, figsize=(15,10), sharey='all')
    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        im = ax[xloc, yloc].imshow(
            efficiencies[ee,...].T,
            vmin=-limit, vmax=limit,
            cmap=coolwarm_cmap
            )

        ax[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax[xloc, yloc].tick_params(labelsize=28)
        ax[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=32
            )

    cax = fig.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar1 = fig.colorbar(
        im, cax=cax, orientation='vertical', ax=ax
    )
    cbar1.set_label(
        r'$\eta$', fontsize=32
        )
    cbar1.ax.tick_params(labelsize=24)
    fig.tight_layout()

    # y-axis label
    fig.text(
        0.025, 0.51,
        r'$\beta \psi_{1}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig.text(
        0.48, 0.03,
        r'$\beta \psi_{2}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.065  # the left side of the subplots of the figure
    right = 0.89    # the right side of the subplots of the figure
    bottom = 0.03   # the bottom of the subplots of the figure
    top = 0.98     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.savefig(
        target_dir
        + "/efficiency_compare_lr_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_nostalgia_scan(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    reference_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )

    nostalgias = zeros((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                prob_ss_array, __, __, force1_array, force2_array = loadtxt(
                    target_dir + reference_file_name.format(
                        E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ), unpack=True
                )

                if (ee==0 and ii==0 and jj==0):
                    N = int(sqrt(prob_ss_array.size))
                    dx = (2*pi)/N
                    positions = linspace(0.0, 2*pi-dx, N)
                    positions_deg = positions * (180.0/npi)

                prob_ss_array = prob_ss_array.reshape((N,N))
                force1_array = force1_array.reshape((N,N))
                force2_array = force2_array.reshape((N,N))

                step_X = empty((N,N))
                step_probability_X(
                    step_X, prob_ss_array, force1_array,
                    m1, gamma, beta, N, dx, 0.001
                    )

                # instantaneous memory
                mem_denom = ((prob_ss_array.sum(axis=1))[:,None]*(prob_ss_array.sum(axis=0))[None,:])
                Imem = (prob_ss_array*log(prob_ss_array/mem_denom)).sum(axis=None)
                # intstantaneous predictive power
                pred_denom = ((step_X.sum(axis=1))[:,None]*(step_X.sum(axis=0))[None,:])
                Ipred = (step_X*log(step_X/pred_denom)).sum(axis=None)

                nostalgias[ee, ii, jj] = Imem - Ipred

    limit=nostalgias.__abs__().max()

    # prepare figure
    fig, ax = subplots(2, 4, figsize=(15,10), sharey='all')
    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        im = ax[xloc, yloc].imshow(
            nostalgias[ee,...].T,
            vmin=-limit, vmax=limit,
            cmap=cm.get_cmap("coolwarm")
            )

        ax[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax[xloc, yloc].tick_params(labelsize=20)
        ax[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=32
            )

    cax = fig.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar1 = fig.colorbar(
        im, cax=cax, orientation='vertical', ax=ax
    )
    cbar1.set_label(
        r'$I_{\mathrm{mem}}-I_{\mathrm{pred}}\ (\mathrm{units\ of\ }\mathrm{nats})$',
        fontsize=32
        )
    cbar1.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar1.ax.tick_params(labelsize=24)
    cbar1.ax.yaxis.offsetText.set_fontsize(24)
    cbar1.ax.yaxis.offsetText.set_x(5.0)
    fig.tight_layout()

    # y-axis label
    fig.text(
        0.025, 0.51,
        r'$\beta \psi_{1}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig.text(
        0.48, 0.03,
        r'$\beta \psi_{2}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.065  # the left side of the subplots of the figure
    right = 0.89    # the right side of the subplots of the figure
    bottom = 0.03   # the bottom of the subplots of the figure
    top = 0.98     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.savefig(
        target_dir
        + "/nostalgia_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_lr_scan(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    reference_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )

    learning_rates = zeros((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                prob_ss_array, __, __, force1_array, force2_array = loadtxt(
                    target_dir + reference_file_name.format(
                        E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ), unpack=True
                )

                if (ee==0 and ii==0 and jj==0):
                    N = int(sqrt(prob_ss_array.size))
                    dx = (2*pi)/N
                    positions = linspace(0.0, 2*pi-dx, N)
                    positions_deg = positions * (180.0/npi)

                prob_ss_array = prob_ss_array.reshape((N,N))
                force1_array = force1_array.reshape((N,N))
                force2_array = force2_array.reshape((N,N))

                flux_array = empty((2,N,N))
                calc_flux(
                    positions, prob_ss_array, force1_array, force2_array,
                    flux_array, m1, m2, gamma, beta, N, dx
                    )

                Dpxgy = empty((N,N))
                calc_derivative_pxgy(
                    prob_ss_array, prob_ss_array.sum(axis=0),
                    Dpxgy,
                    N, dx
                )

                learning = flux_array[1,...]*Dpxgy

                learning_rates[ee, ii, jj] = trapz(
                    trapz(learning, dx=dx, axis=1), dx=dx
                    )

    limit=learning_rates.__abs__().max()

    # prepare figure
    fig, ax = subplots(2, 4, figsize=(15,10), sharey='all')
    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        im = ax[xloc, yloc].imshow(
            learning_rates[ee,...].T,
            vmin=-limit, vmax=limit,
            cmap=cm.get_cmap("coolwarm")
            )

        ax[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax[xloc, yloc].tick_params(labelsize=20)
        ax[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=32
            )

    cax = fig.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar1 = fig.colorbar(
        im, cax=cax, orientation='vertical',
        ax=ax
    )
    cbar1.set_label(
        r'$l_{\mathrm{sys}}\ (\mathrm{units\ of\ }\mathrm{nats}\cdot\mathrm{s}^{-1})$',
        fontsize=32
        )
    cbar1.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar1.ax.tick_params(labelsize=24)
    cbar1.ax.yaxis.offsetText.set_fontsize(24)
    cbar1.ax.yaxis.offsetText.set_x(5.0)
    fig.tight_layout()

    # y-axis label
    fig.text(
        0.025, 0.51,
        r'$\beta \psi_{1}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig.text(
        0.48, 0.03,
        # r'$\beta F_{\mathrm{ATP}}$',
        r'$\beta \psi_{2}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.065  # the left side of the subplots of the figure
    right = 0.89    # the right side of the subplots of the figure
    bottom = 0.03   # the bottom of the subplots of the figure
    top = 0.98     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.savefig(
        target_dir
        + "/learning_rate_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_lr_efficiency_correlation(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    reference_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )
    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )


    learning_rates = empty((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))
    efficiencies = empty((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                prob_ss_array, force1_array, force2_array = loadtxt(
                    target_dir + reference_file_name.format(
                        E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ), unpack=True, usecols=(0,3,4)
                )
                Ecouple_array_out, efficiency_ratio = loadtxt(
                    target_dir + input_file_name.format(
                        E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ),
                    unpack=True, usecols=(0,5)
                )

                if (ee==0 and ii==0 and jj==0):
                    N = int(sqrt(prob_ss_array.size))
                    dx = (2*pi)/N
                    positions = linspace(0.0, 2*pi-dx, N)
                    positions_deg = positions * (180.0/npi)


                loc=where(
                    (Ecouple_array_out-Ecouple).__abs__()<=finfo('float32').eps
                    )[0][0]

                prob_ss_array = prob_ss_array.reshape((N,N))
                force1_array = force1_array.reshape((N,N))
                force2_array = force2_array.reshape((N,N))

                flux_array = empty((2,N,N))
                calc_flux(
                    positions, prob_ss_array, force1_array, force2_array,
                    flux_array, m1, m2, gamma, beta, N, dx
                    )

                Dpxgy = empty((N,N))
                calc_derivative_pxgy(
                    prob_ss_array, prob_ss_array.sum(axis=0),
                    Dpxgy,
                    N, dx
                )

                learning = flux_array[1,...]*Dpxgy

                if (abs(F_atp)==abs(F_Hplus)):
                    efficiencies[ee, ii, jj] = nan
                else:
                    efficiencies[ee, ii, jj] = efficiency_ratio[loc]

                learning_rates[ee, ii, jj] = learning.sum(axis=None)

    # prepare figure
    fig, ax = subplots(2, 4, figsize=(15,10), sharey='all', sharex='all')
    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        corr_array = correlate2d(efficiencies[ee,...], learning_rates[ee,...])

        im = ax[xloc, yloc].imshow(
            corr_array.T,
            vmin=-1.0, vmax=1.0,
            cmap=cm.get_cmap("coolwarm")
            )

        ax[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax[xloc, yloc].set_xticklabels(F_atp_array)
        ax[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax[xloc, yloc].set_yticklabels(F_Hplus_array)
        ax[xloc, yloc].tick_params(labelsize=20)
        ax[xloc, yloc].set_title(r"$\beta E_{\mathrm{couple}}$ = " + str(Ecouple), fontsize=24)

        if (yloc==0): ax[xloc, yloc].set_ylabel(r"$\beta\mathrm{F}_{\mathrm{H}^{+}}$", fontsize=28)
        if (xloc==1): ax[xloc, yloc].set_xlabel(r"$\beta\mathrm{F}_{\mathrm{atp}}$", fontsize=28)

        divider = make_axes_locatable(ax[xloc, yloc])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig.colorbar(im, ax=ax[xloc, yloc], cax=cax)
        cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig(
        target_dir
        + "/correlation_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_lr_efficiency_scatter(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    reference_file_name = (
        "reference_"
        + "E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}"
        + "_outfile.dat"
        )
    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    learning_rates = empty((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))
    efficiencies = empty((Ecouple_array.size, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                prob_ss_array, force1_array, force2_array = loadtxt(
                    target_dir + reference_file_name.format(
                        E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ), unpack=True, usecols=(0,3,4)
                )
                Ecouple_array_out, efficiency_ratio = loadtxt(
                    target_dir + input_file_name.format(
                        E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ),
                    unpack=True, usecols=(0,5)
                )

                if (ee==0 and ii==0 and jj==0):
                    N = int(sqrt(prob_ss_array.size))
                    dx = (2*pi)/N
                    positions = linspace(0.0, 2*pi-dx, N)
                    positions_deg = positions * (180.0/npi)

                loc=where(
                    (Ecouple_array_out-Ecouple).__abs__()<=finfo('float32').eps
                    )[0][0]

                prob_ss_array = prob_ss_array.reshape((N,N))
                force1_array = force1_array.reshape((N,N))
                force2_array = force2_array.reshape((N,N))

                flux_array = empty((2,N,N))
                calc_flux(
                    positions, prob_ss_array, force1_array, force2_array,
                    flux_array, m1, m2, gamma, beta, N, dx
                    )

                Dpxgy = empty((N,N))
                calc_derivative_pxgy(
                    prob_ss_array, prob_ss_array.sum(axis=0),
                    Dpxgy,
                    N, dx
                )

                learning = flux_array[1,...]*Dpxgy

                if (abs(F_atp)==abs(F_Hplus)):
                    efficiencies[ee, ii, jj] = nan
                else:
                    efficiencies[ee, ii, jj] = efficiency_ratio[loc]

                learning_rates[ee, ii, jj] = learning.sum(axis=None)


    # prepare figure
    fig, ax = subplots(1, 1, figsize=(10,10), sharey='all', sharex='all')

    flat_efficiencies = efficiencies.flatten()

    im = ax.scatter(
        learning_rates.flatten()[~(isnan(flat_efficiencies))], flat_efficiencies[~(isnan(flat_efficiencies))],
        color="black", lw=3.0
        )

    ax.tick_params(labelsize=20)
    ax.set_xlim([-8.0e-6, 8.0e-6])
    # ax.set_ylim([-0.5, 0.5])
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_title(r"$\beta E_{\mathrm{couple}}$ = " + str(Ecouple), fontsize=24)

    ax.set_ylabel(r"$\eta$", fontsize=28)
    ax.set_xlabel(r"$l_{x_{2}}$", fontsize=28)

    fig.tight_layout()
    fig.savefig(
        target_dir
        + "/scatter_efficiency_lr_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_flux_lr_scan(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    fluxes = zeros((Ecouple_array.size, 2, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                Ecouple_array_out, integrate_flux_X, integrate_flux_Y = loadtxt(
                    target_dir + input_file_name.format(
                        E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ),
                    unpack=True, usecols=(0,1,2)
                )

                loc=where((Ecouple_array_out-Ecouple).__abs__()<=finfo('float32').eps)[0][0]
                fluxes[ee, 0, ii, jj] = integrate_flux_X[loc]
                fluxes[ee, 1, ii, jj] = integrate_flux_Y[loc]

    limit=fluxes[~(isnan(fluxes))].__abs__().max()

    # prepare figures
    fig1, ax1 = subplots(2, 4, figsize=(15,10), sharey='all')
    fig2, ax2 = subplots(2, 4, figsize=(15,10), sharey='all')

    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        im1 = ax1[xloc, yloc].imshow(
            fluxes[ee, 0, ...].T,
            vmin=-limit, vmax=limit,
            cmap=cm.get_cmap("coolwarm")
            )
        im2 = ax2[xloc, yloc].imshow(
            fluxes[ee, 1, ...].T,
            vmin=-limit, vmax=limit,
            cmap=cm.get_cmap("coolwarm")
            )

        ax1[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax1[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax1[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax1[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax1[xloc, yloc].tick_params(labelsize=22)
        ax1[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=32
            )

        ax2[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax2[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax2[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax2[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax2[xloc, yloc].tick_params(labelsize=22)
        ax2[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=32
            )

    cbar_ticks = array([-5.0, -2.5, 0.0, 2.5, 5.0])*1e-3

    cax1 = fig1.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar1 = fig1.colorbar(
        im1, cax=cax1, orientation='vertical', ax=ax1
    )
    cbar1.set_label(
        r'$\mathcal{J}_{1}^{\mathrm{int}}\ (\mathrm{units\ of\ }\mathrm{s}^{-1})$', fontsize=32
        )
    cbar1.set_ticks(cbar_ticks)
    cbar1.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar1.ax.tick_params(labelsize=24)
    cbar1.ax.yaxis.offsetText.set_fontsize(24)
    cbar1.ax.yaxis.offsetText.set_x(5.0)
    cbar1.ax.yaxis.offsetText.set_y(5.0)
    fig1.tight_layout()

    # y-axis label
    fig1.text(
        0.025, 0.51,
        r'$\beta \psi_{1}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig1.text(
        0.48, 0.03,
        r'$\beta \psi_{2}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, va='center', ha='center'
    )

    left = 0.065  # the left side of the subplots of the figure
    right = 0.89    # the right side of the subplots of the figure
    bottom = 0.03   # the bottom of the subplots of the figure
    top = 0.98     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig1.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    cax2 = fig2.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar2 = fig2.colorbar(
        im2, cax=cax2, orientation='vertical', ax=ax2
    )
    cbar2.set_label(
        r'$\mathcal{J}_{2}^{\mathrm{int}}\ (\mathrm{units\ of\ }\mathrm{s}^{-1})$', fontsize=32
        )
    cbar2.set_ticks(cbar_ticks)
    cbar2.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar2.ax.tick_params(labelsize=24)
    cbar2.ax.yaxis.offsetText.set_fontsize(24)
    cbar2.ax.yaxis.offsetText.set_x(5.0)
    fig2.tight_layout()

    # y-axis label
    fig2.text(
        0.025, 0.51,
        # r'$\beta F_{\mathrm{H}^{+}}$',
        r'$\beta \psi_{1}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig2.text(
        0.48, 0.03,
        # r'$\beta F_{\mathrm{ATP}}$',
        r'$\beta \psi_{2}\ (\mathrm{units\ of\ }\mathrm{rad}^{-1})$',
        fontsize=36, va='center', ha='center'
    )

    fig2.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig1.savefig(
        target_dir
        + "/flux1_compare_lr_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )
    fig2.savefig(
        target_dir
        + "/flux2_compare_lr_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_power_lr_scan(target_dir):

    [
        __, E0, Ecouple, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    input_file_name = (
        "/flux_power_efficiency_"
        + "E0_{0}_E1_{1}_F_Hplus_{2}_F_atp_{3}_minima_{4}_phase_{5}"
        + "_outfile.dat"
        )

    powers = zeros((Ecouple_array.size, 2, F_atp_array.size,F_Hplus_array.size))

    for ee, Ecouple in enumerate(Ecouple_array):
        for ii, F_atp in enumerate(F_atp_array):
            for jj, F_Hplus in enumerate(F_Hplus_array):

                Ecouple_array_out, integrate_power_X, integrate_power_Y = loadtxt(
                    target_dir + input_file_name.format(
                        E0, E1, F_Hplus, F_atp, num_minima, phase_shift
                        ),
                    unpack=True, usecols=(0,3,4)
                )

                loc=where((Ecouple_array_out-Ecouple).__abs__()<=finfo('float32').eps)[0][0]
                powers[ee, 0, ii, jj] = integrate_power_X[loc]
                powers[ee, 1, ii, jj] = integrate_power_Y[loc]

    limit=powers[~(isnan(powers))].__abs__().max()

    # prepare figures
    fig1, ax1 = subplots(2, 4, figsize=(15,10), sharey='all')
    fig2, ax2 = subplots(2, 4, figsize=(15,10), sharey='all')

    for ee, Ecouple in enumerate(Ecouple_array):

        xloc, yloc = ee//4, ee%4

        im1 = ax1[xloc, yloc].imshow(
            powers[ee, 0, ...].T,
            vmin=-limit, vmax=limit,
            cmap=cm.get_cmap("coolwarm")
            )
        im2 = ax2[xloc, yloc].imshow(
            powers[ee, 1, ...].T,
            vmin=-limit, vmax=limit,
            cmap=cm.get_cmap("coolwarm")
            )

        ax1[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax1[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax1[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax1[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax1[xloc, yloc].tick_params(labelsize=20)
        ax1[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=22
            )

        ax2[xloc, yloc].set_xticks(list(range(F_atp_array.size)))
        ax2[xloc, yloc].set_xticklabels(F_atp_array.astype(int))
        ax2[xloc, yloc].set_yticks(list(range(F_Hplus_array.size)))
        ax2[xloc, yloc].set_yticklabels(F_Hplus_array.astype(int))
        ax2[xloc, yloc].tick_params(labelsize=20)
        ax2[xloc, yloc].set_title(
            r"$\beta E_{\mathrm{couple}}=$" + " {0}".format(int(Ecouple)),
            fontsize=22
            )

    cax1 = fig1.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar1 = fig1.colorbar(
        im1, cax=cax1, orientation='vertical',
        ax=ax1
    )
    cbar1.set_label(
        r'$\mathcal{P}_{1}$',
        fontsize=26
        )
    cbar1.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar1.ax.tick_params(labelsize=20)
    cbar1.ax.yaxis.offsetText.set_fontsize(18)
    cbar1.ax.yaxis.offsetText.set_x(4.0)
    fig1.tight_layout()

    # y-axis label
    fig1.text(
        0.02, 0.51,
        # r'$\beta F_{\mathrm{H}^{+}}$',
        r'$\beta \psi_{1}$',
        fontsize=28, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig1.text(
        0.48, 0.03,
        # r'$\beta F_{\mathrm{ATP}}$',
        r'$\beta \psi_{2}$',
        fontsize=28, va='center', ha='center'
    )

    left = 0.065  # the left side of the subplots of the figure
    right = 0.89    # the right side of the subplots of the figure
    bottom = 0.03   # the bottom of the subplots of the figure
    top = 0.98     # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    fig1.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    cax2 = fig2.add_axes([0.90, 0.12, 0.01, 0.77])
    cbar2 = fig2.colorbar(
        im2, cax=cax2, orientation='vertical',
        ax=ax2
    )
    cbar2.set_label(
        r'$\mathcal{P}_{2}$',
        fontsize=26
        )
    cbar2.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar2.ax.tick_params(labelsize=20)
    cbar2.ax.yaxis.offsetText.set_fontsize(18)
    cbar2.ax.yaxis.offsetText.set_x(4.0)
    fig2.tight_layout()

    # y-axis label
    fig2.text(
        0.02, 0.51,
        # r'$\beta F_{\mathrm{H}^{+}}$',
        r'$\beta \psi_{1}$',
        fontsize=28, rotation='vertical', va='center', ha='center'
    )
    # x-axis label
    fig2.text(
        0.48, 0.03,
        # r'$\beta F_{\mathrm{ATP}}$',
        r'$\beta \psi_{2}$',
        fontsize=28, va='center', ha='center'
    )

    fig2.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig1.savefig(
        target_dir
        + "/power1_compare_lr_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )
    fig2.savefig(
        target_dir
        + "/power2_compare_lr_scan_E0_{0}_E1_{1}_minima_{2}_phase_{3}".format(
                E0, E1, num_minima, phase_shift
            )
        + "_figure.pdf"
        )

def plot_emma_compare(target_dir):

    [
        N, E0, __, E1, __, __, num_minima, phase_shift,
        m1, m2, beta, gamma
        ] = set_params()

    emma_file_template = "Emma_Flux_Ecouple_Fx_{0}_Fy{1}.dat"
    my_file_template = (
        "flux_power_efficiency_" 
        + "E0_0.0_E1_0.0_F_Hplus_{0}_F_atp_{1}_minima_{2}_phase_{3}_outfile.dat"
    )

    F_atp_vals = [-2.0, -4.0]
    F_Hplus_vals = [2.0, 4.0]

    dx = (2*pi)/N

    fig, ax = subplots(
        len(F_atp_vals), len(F_Hplus_vals), figsize=(10,10), 
        sharey='row', sharex='col'
        )

    for i, F_atp in enumerate(F_atp_vals):
        for j, F_Hplus in enumerate(F_Hplus_vals):

            try:
                e_Ecouple, e_Jx, e_Jy = loadtxt(
                    target_dir + emma_file_template.format(F_Hplus, F_atp), unpack=True
                    )
                j_Ecouple, j_Jx, j_Jy = loadtxt(
                    target_dir + 
                    my_file_template.format(F_Hplus, F_atp, num_minima, phase_shift),
                    usecols=(0,1,2), unpack=True
                )

                e_Jx_reduced = empty(j_Ecouple.size)
                e_Jy_reduced = empty(j_Ecouple.size)
                counter = 0

                for ii, Ecouple in enumerate(e_Ecouple):
                    if Ecouple in j_Ecouple:
                        e_Jx_reduced[counter] = e_Jx[ii]
                        e_Jy_reduced[counter] = e_Jy[ii]
                        counter += 1

                ax[i, j].plot(
                    j_Ecouple[:counter], e_Jx_reduced[:counter], 'k-', 
                    j_Ecouple[:counter], e_Jy_reduced[:counter], 'r-', 
                    lw=3.0
                    )
                ax[i, j].plot(j_Ecouple, j_Jx, 'k--', j_Ecouple, j_Jy, 'r--', lw=3.0)
            except OSError:
                continue
            
            ax[i, j].set_xlim([0.0, 40.0])

    fig.tight_layout()
    fig.savefig(target_dir + "Joseph_Emma_compare_figure.pdf")


if __name__ == "__main__":
    target_dir = "/Users/jlucero/data_dir/2019-04-09/"
    # target_dir = "./"
    # calculate_flux_power_and_efficiency(target_dir)
    # plot_energy(target_dir)
    # plot_probability(target_dir)
    # plot_probability_against_reference(ref_dir, target_dir)
    # plot_power(target_dir)
    # plot_efficiency(target_dir)
    # plot_efficiency_against_ratio(target_dir)
    # plot_flux(target_dir)
    # plot_lr(target_dir)
    # plot_energy_scan(target_dir)
    # plot_probability_eq_scan(target_dir)
    # plot_probability_scan(target_dir)
    # plot_flux_scan(target_dir)
    # plot_integrated_flux_scan(target_dir)
    # plot_power_scan(target_dir)
    # plot_efficiency_scan(target_dir)
    # plot_efficiency_scan_compare(target_dir)
    # plot_efficiencies_lr_scan(target_dir)
    # plot_relative_entropy_lr_scan(target_dir)
    # plot_nostalgia_scan(target_dir)
    # plot_lr_scan(target_dir)
    # plot_lr_efficiency_correlation(target_dir)
    # plot_lr_efficiency_scatter(target_dir)
    # plot_flux_lr_scan(target_dir)
    # plot_power_lr_scan(target_dir)
    plot_emma_compare(target_dir)
