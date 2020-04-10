from numpy import array, linspace, loadtxt, append, pi, empty, sqrt, zeros, asarray, trapz
import math
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

N = 360  # N x N grid is used for Fokker-Planck simulations
dx = 2 * math.pi / N  # spacing between gridpoints
positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
timescale = 1.5 * 10**4  # conversion factor between simulation and experimental timescale

E0 = 2.0  # barrier height Fo
E1 = 2.0  # barrier height F1
psi_1 = 4.0  # chemical driving force on Fo
psi_2 = -2.0  # chemical driving force on F1
num_minima1 = 3.0  # number of barriers in Fo's landscape
num_minima2 = 3.0  # number of barriers in F1's landscape

Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])  # coupling strengths
min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])  # number of energy minima/ barriers

def calc_flux(p_now, drift_at_pos, diffusion_at_pos, flux_array, N):
    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] = (
        (drift_at_pos[0, 0, 0]*p_now[0, 0])
        -(diffusion_at_pos[0, 1, 0]*p_now[1, 0]-diffusion_at_pos[0, N-1, 0]*p_now[N-1, 0])/(2.0*dx)
        -(diffusion_at_pos[1, 0, 1]*p_now[0, 1]-diffusion_at_pos[1, 0, N-1]*p_now[0, N-1])/(2.0*dx)
        )
    flux_array[0, 0, N-1] = (
        (drift_at_pos[0, 0, N-1]*p_now[0, N-1])
        -(diffusion_at_pos[0, 1, N-1]*p_now[1, N-1]-diffusion_at_pos[0, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        -(diffusion_at_pos[1, 0, 0]*p_now[0, 0]-diffusion_at_pos[1, 0, N-2]*p_now[0, N-2])/(2.0*dx)
        )
    flux_array[0, N-1, 0] = (
        (drift_at_pos[0, N-1, 0]*p_now[N-1, 0])
        -(diffusion_at_pos[0, 0, 0]*p_now[0, 0]-diffusion_at_pos[0, N-2, 0]*p_now[N-2, 0])/(2.0*dx)
        -(diffusion_at_pos[1, N-1, 1]*p_now[N-1, 1]-diffusion_at_pos[1, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        )
    flux_array[0, N-1, N-1] = (
        (drift_at_pos[0, N-1, N-1]*p_now[N-1, N-1])
        -(diffusion_at_pos[0, 0, N-1]*p_now[0, N-1]-diffusion_at_pos[0, N-2, N-1]*p_now[N-2, N-1])/(2.0*dx)
        -(diffusion_at_pos[1, N-1, 0]*p_now[N-1, 0]-diffusion_at_pos[1, N-1, N-2]*p_now[N-1, N-2])/(2.0*dx)
        )

    # second component
    flux_array[1, 0, 0] = (
        (drift_at_pos[1, 0, 0]*p_now[0, 0])
        -(diffusion_at_pos[2, 1, 0]*p_now[1, 0]-diffusion_at_pos[2, N-1, 0]*p_now[N-1, 0])/(2.0*dx)
        -(diffusion_at_pos[3, 0, 1]*p_now[0, 1]-diffusion_at_pos[3, 0, N-1]*p_now[0, N-1])/(2.0*dx)
        )
    flux_array[1, 0, N-1] = (
        (drift_at_pos[1, 0, N-1]*p_now[0, N-1])
        -(diffusion_at_pos[2, 1, N-1]*p_now[1, N-1]-diffusion_at_pos[2, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        -(diffusion_at_pos[3, 0, 0]*p_now[0, 0]-diffusion_at_pos[3, 0, N-2]*p_now[0, N-2])/(2.0*dx)
        )
    flux_array[1, N-1, 0] = (
        (drift_at_pos[1, N-1, 0]*p_now[N-1, 0])
        -(diffusion_at_pos[2, 0, 0]*p_now[0, 0]-diffusion_at_pos[2, N-2, 0]*p_now[N-2, 0])/(2.0*dx)
        -(diffusion_at_pos[3, N-1, 1]*p_now[N-1, 1]-diffusion_at_pos[3, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        )
    flux_array[1, N-1, N-1] = (
        (drift_at_pos[1, N-1, N-1]*p_now[N-1, N-1])
        -(diffusion_at_pos[2, 0, N-1]*p_now[0, N-1]-diffusion_at_pos[2, N-2, N-1]*p_now[N-2, N-1])/(2.0*dx)
        -(diffusion_at_pos[3, N-1, 0]*p_now[N-1, 0]-diffusion_at_pos[3, N-1, N-2]*p_now[N-1, N-2])/(2.0*dx)
        )

    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] = (
            (drift_at_pos[0, 0, i]*p_now[0, i])
            -(diffusion_at_pos[0, 1, i]*p_now[1, i]-diffusion_at_pos[0, N-1, i]*p_now[N-1, i])/(2.0*dx)
            -(diffusion_at_pos[1, 0, i+1]*p_now[0, i+1]-diffusion_at_pos[1, 0, i-1]*p_now[0, i-1])/(2.0*dx)
            )
        flux_array[0, i, 0] = (
            (drift_at_pos[0, i, 0]*p_now[i, 0])
            -(diffusion_at_pos[0, i+1, 0]*p_now[i+1, 0]-diffusion_at_pos[0, i-1, 0]*p_now[i-1, 0])/(2.0*dx)
            -(diffusion_at_pos[1, i, 1]*p_now[i, 1]-diffusion_at_pos[1, i, N-1]*p_now[i, N-1])/(2.0*dx)
            )
        flux_array[0, N-1, i] = (
            (drift_at_pos[0, N-1, i]*p_now[N-1, i])
            -(diffusion_at_pos[0, 0, i]*p_now[0, i]-diffusion_at_pos[0, N-2, i]*p_now[N-2, i])/(2.0*dx)
            -(diffusion_at_pos[1, N-1, i+1]*p_now[N-1, i+1]-diffusion_at_pos[1, N-1, i-1]*p_now[N-1, i-1])/(2.0*dx)
            )
        flux_array[0, i, N-1] = (
            (drift_at_pos[0, i, N-1]*p_now[i, N-1])
            -(diffusion_at_pos[0, i+1, N-1]*p_now[i+1, N-1]-diffusion_at_pos[0, i-1, N-1]*p_now[i-1, N-1])/(2.0*dx)
            -(diffusion_at_pos[1, i, 0]*p_now[i, 0]-diffusion_at_pos[1, i, N-2]*p_now[i, N-2])/(2.0*dx)
            )

        # second component
        flux_array[1, 0, i] = (
            (drift_at_pos[1, 0, i]*p_now[0, i])
            -(diffusion_at_pos[2, 1, i]*p_now[1, i]-diffusion_at_pos[2, N-1, i]*p_now[N-1, i])/(2.0*dx)
            -(diffusion_at_pos[3, 0, i+1]*p_now[0, i+1]-diffusion_at_pos[3, 0, i-1]*p_now[0, i-1])/(2.0*dx)
            )
        flux_array[1, i, 0] = (
            (drift_at_pos[1, i, 0]*p_now[i, 0])
            -(diffusion_at_pos[2, i+1, 0]*p_now[i+1, 0]-diffusion_at_pos[2, i-1, 0]*p_now[i-1, 0])/(2.0*dx)
            -(diffusion_at_pos[3, i, 1]*p_now[i, 1]-diffusion_at_pos[3, i, N-1]*p_now[i, N-1])/(2.0*dx)
            )
        flux_array[1, N-1, i] = (
            (drift_at_pos[1, N-1, i]*p_now[N-1, i])
            -(diffusion_at_pos[2, 0, i]*p_now[0, i]-diffusion_at_pos[2, N-2, i]*p_now[N-2, i])/(2.0*dx)
            -(diffusion_at_pos[3, N-1, i+1]*p_now[N-1, i+1]-diffusion_at_pos[3, N-1, i-1]*p_now[N-1, i-1])/(2.0*dx)
            )
        flux_array[1, i, N-1] = (
            (drift_at_pos[1, i, N-1]*p_now[i, N-1])
            -(diffusion_at_pos[2, i+1, N-1]*p_now[i+1, N-1]-diffusion_at_pos[2, i-1, N-1]*p_now[i-1, N-1])/(2.0*dx)
            -(diffusion_at_pos[3, i, 0]*p_now[i, 0]-diffusion_at_pos[3, i, N-2]*p_now[i, N-2])/(2.0*dx)
            )

        # for points with well defined neighbours
        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] = (
                (drift_at_pos[0, i, j]*p_now[i, j])
                -(diffusion_at_pos[0, i+1, j]*p_now[i+1, j]-diffusion_at_pos[0, i-1, j]*p_now[i-1, j])/(2.0*dx)
                -(diffusion_at_pos[1, i, j+1]*p_now[i, j+1]-diffusion_at_pos[1, i, j-1]*p_now[i, j-1])/(2.0*dx)
                )
            # second component
            flux_array[1, i, j] = (
                (drift_at_pos[1, i, j]*p_now[i, j])
                -(diffusion_at_pos[2, i+1, j]*p_now[i+1, j]-diffusion_at_pos[2, i-1, j]*p_now[i-1, j])/(2.0*dx)
                -(diffusion_at_pos[3, i, j+1]*p_now[i, j+1]-diffusion_at_pos[3, i, j-1]*p_now[i, j-1])/(2.0*dx)
                )

def flux_power_efficiency(target_dir): # processing of raw data
    phase_array = array([0.0])
    psi1_array = array([4.0])
    psi2_array = array([-2.0])

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            integrate_flux_X = empty(phase_array.size)
            integrate_flux_Y = empty(phase_array.size)
            integrate_power_X = empty(phase_array.size)
            integrate_power_Y = empty(phase_array.size)
            efficiency_ratio = empty(phase_array.size)

            for Ecouple in Ecouple_array:
                for ii, phase_shift in enumerate(phase_array):
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")

                    output_file_name = (target_dir + "flux_power_efficiency_" +
                                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")

                    print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " +
                          f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")

                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1,
                                                                    num_minima2, phase_shift),
                                             usecols=(0, 3, 4, 5, 6, 7, 8))
                        N = int(sqrt(len(data_array)))  # check grid size
                        print('Grid size: ', N)

                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

                        flux_array = zeros((2, N, N))
                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array)/(dx*dx)

                        # Note that the factor of 2 pi actually needs to be removed to get the right units.
                        # Currently, all the powers being plotted in this script are multiplied by 2 pi
                        # to make up for this factor
                        integrate_flux_X[ii] = (1/(2*pi))*trapz(trapz(flux_array[0, ...], dx=dx, axis=1), dx=dx)
                        integrate_flux_Y[ii] = (1/(2*pi))*trapz(trapz(flux_array[1, ...], dx=dx, axis=0), dx=dx)

                        integrate_power_X[ii] = integrate_flux_X[ii]*psi_1
                        integrate_power_Y[ii] = integrate_flux_Y[ii]*psi_2
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                     phase_shift))
                if abs(psi_1) <= abs(psi_2):
                    efficiency_ratio = -(integrate_power_X/integrate_power_Y)
                else:
                    efficiency_ratio = -(integrate_power_Y/integrate_power_X)

                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), "w") as \
                        ofile:
                    for ii, phase_shift in enumerate(phase_array):
                        ofile.write(
                            f"{phase_shift:.15e}" + "\t"
                            + f"{integrate_flux_X[ii]:.15e}" + "\t"
                            + f"{integrate_flux_Y[ii]:.15e}" + "\t"
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{efficiency_ratio[ii]:.15e}" + "\n")
                    ofile.flush()

def plot_power_efficiency_Ecouple(target_dir):  # plot power and efficiency vs coupling strength
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0,
         45.25, 64.0, 90.51, 128.0])

    output_file_name = (
            target_dir + "P_ATP_eff_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 8))

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis
    maxpower = 2 * pi * 0.000085247 * timescale
    axarr[0].axhline(maxpower, color='black', linestyle=':', linewidth=1)  # line at infinite power coupling
    axarr[0].axvline(12, color='black', linestyle='--', linewidth=1)  # lining up features in the two plots
    axarr[0].fill_between([1, 250], 0, 31, facecolor='grey', alpha=0.2)  # shading power output

    # zero-barrier results
    input_file_name = (target_dir + "plotting_data/"
                       + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    Ecouple_array2 = array(data_array[:, 0])
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', color='C0', label='$0$', linewidth=2)

    # Fokker-Planck results (barriers)
    i = 0  # only use phase=0 data
    power_y_array = []
    for ii, Ecouple in enumerate(Ecouple_array_tot):
        input_file_name = (target_dir + "plotting_data/" + "flux_power_efficiency_"
                           + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            if Ecouple in Ecouple_array:  # data format varies a little
                power_y = array(data_array[i, 1])
            else:
                power_y = array(data_array[1])
            power_y_array = append(power_y_array, power_y)
        except OSError:
            print('Missing file flux')
    axarr[0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, 'o', color='C1', label='$2$', markersize=8)

    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].spines['bottom'].set_visible(False)
    axarr[0].set_xlim((1.7, 135))
    axarr[0].set_ylim((-60, 31))
    axarr[0].set_yticks([-50, -25, 0, 25])

    leg = axarr[0].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    #####################################################
    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x axis
    axarr[1].axvline(12, color='black', linestyle='--', linewidth=1)  # lining up features
    axarr[1].axhline(1, color='black', linestyle=':', linewidth=1)  # max efficiency
    axarr[1].fill_between([1, 250], 0, 1, facecolor='grey', alpha=0.2)  # shading power output

    # zero-barrier curve
    input_file_name = (
            target_dir + "plotting_data/"
            + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    try:
        data_array = loadtxt(input_file_name.format(psi_1, psi_2))
        Ecouple_array2 = array(data_array[1:, 0])
        Ecouple_array2 = append(Ecouple_array2, 128.0)  # add point to end up with curves of equal length
        flux_x_array = array(data_array[1:, 1])
        flux_y_array = array(data_array[1:, 2])  # skip the point at zero, which is problematic on a log scale
        flux_x_array = append(flux_x_array, flux_x_array[-1])  # copy last point to add one
        flux_y_array = append(flux_y_array, flux_y_array[-1])
        axarr[1].plot(Ecouple_array2, flux_y_array / (flux_x_array), '-', color='C0', linewidth=2)
    except:
        print('Missing data efficiency')

    # Fokker-Planck results (barriers)
    eff_array = []
    for ii, Ecouple in enumerate(Ecouple_array_tot):
        input_file_name = (
                target_dir + "plotting_data/flux_power_efficiency_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=5)
            if Ecouple in Ecouple_array:
                eff_array = append(eff_array, data_array[0])
            else:
                eff_array = append(eff_array, data_array)
        except OSError:
            print('Missing file efficiency')
    axarr[1].plot(Ecouple_array_tot, eff_array / (-psi_2 / psi_1), 'o', color='C1', markersize=8)

    axarr[1].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1].set_xscale('log')
    axarr[1].set_xlim((1.7, 135))
    axarr[1].set_ylim((-0.5, 1.05))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].spines['bottom'].set_visible(False)
    axarr[1].set_yticks([-0.5, 0, 0.5, 1.0])
    axarr[1].tick_params(axis='both', labelsize=14)

    f.text(0.05, 0.95, r'$\mathbf{a)}$', ha='center', fontsize=20)
    f.text(0.05, 0.48, r'$\mathbf{b)}$', ha='center', fontsize=20)
    f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_power_Ecouple_grid(target_dir):  # grid of plots of the flux as a function of the phase offset
    Ecouple_array_tot = array([5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])

    output_file_name = (target_dir + "P_ATP_Ecouple_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='row', figsize=(8, 6))
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print('Chemical driving forces:', psi_1, psi_2)

            # line at highest Ecouple power
            input_file_name = (
                            target_dir + "plotting_data/"
                            + "Power_Ecouple_inf_grid_E0_2.0_E1_2.0_n1_3.0_n2_3.0_outfile.dat")
            try:
                inf_array = loadtxt(input_file_name, usecols=2)
            except OSError:
                print('Missing file Infinite Power Coupling')

            axarr[i, j].axhline(2*pi*inf_array[i*3 + j] * timescale, color='grey', linestyle=':', linewidth=1)

            # zero-barrier result
            input_file_name = (
                        target_dir + "plotting_data/"
                        + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2

            axarr[i, j].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', color='C0', linewidth=3)

            # Fokker-Planck results (barriers)
            power_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (
                            target_dir + "plotting_data/" + "flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=4)

                    if data_array.size > 2:  # data format varies a little
                        power_y = array(data_array[0])
                    else:
                        power_y = array(data_array)
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file flux')
            axarr[i, j].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, '.', color='C1', markersize=14)

            # print('Ratio max power / infinite coupling power', amax(-power_y_array)/inf_array[i*3 + j], '\n')

            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((1.6, 150))
            elif j == 1:
                axarr[i, j].set_xlim((2.3, 150))
            else:
                axarr[i, j].set_xlim((5, 150))

            if i == 0:
                axarr[i, j].set_ylim((0, 7.8))
                axarr[i, j].set_yticks([0, 3.0, 6.0])
                axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            elif i == 1:
                axarr[i, j].set_ylim((0, 31))
                axarr[i, j].set_yticks([0, 15, 30])
                axarr[i, j].set_yticklabels([r'$0$', r'$15$', r'$30$'])
            else:
                axarr[i, j].set_ylim((0, 122))
                axarr[i, j].set_yticks([0, 50, 100])
                axarr[i, j].set_yticklabels([r'$0$', r'$50$', r'$100$'])

            if j == 0 and i > 0:
                axarr[i, j].yaxis.offsetText.set_fontsize(0)
            else:
                axarr[i, j].yaxis.offsetText.set_fontsize(14)

            if j == psi1_array.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi_ratio[::-1][i], labelpad=16, rotation=270, fontsize=18)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.0f$' % psi1_array[::-1][j], fontsize=18)

            if j == 2 and i == 1:
                axarr[i, j].tick_params(axis='x', colors='red', which='both')
                axarr[i, j].tick_params(axis='y', colors='red', which='both')
                axarr[i, j].spines['left'].set_color('red')
                axarr[i, j].spines['bottom'].set_color('red')
            else:
                axarr[i, j].tick_params(axis='both', labelsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=24)
    f.text(0.01, 0.5, r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', va='center', rotation='vertical',
           fontsize=24)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=24)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (k_{\rm B} T / \rm rad)$', va='center', rotation=270, fontsize=24)

    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))

def plot_power_efficiency_phi(target_dir): # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "power_efficiency_phi_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 6))

    # flux plot
    axarr[0].axhline(0, color='black', linewidth=1)  # line at zero

    # zero-barrier results
    input_file_name = (
                target_dir + "plotting_data/"
                + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].axhline(2*pi*power_y[28]*timescale, color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    for ii, Ecouple in enumerate([16.0]):
        input_file_name = (
                    target_dir + "plotting_data/" + "flux_power_efficiency_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            phase_array = array(data_array[:, 0])
            power_y = array(data_array[:, 1])
        except OSError:
            print('Missing file flux')
    axarr[0].plot(phase_array, -2*pi*power_y*timescale, 'o', color='C1', label='$2$', markersize=8)

    axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='both', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].set_ylim((0, 31))
    axarr[0].set_xlim((0, 2.1))
    axarr[0].set_yticks([0, 10, 20, 30])

    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x-axis
    axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')  # max efficiency
    axarr[1].set_aspect(0.5)

    for ii, Ecouple in enumerate([16.0]):
        input_file_name = (
                    target_dir + "plotting_data/flux_power_efficiency_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=5)
            eff_array = data_array
        except OSError:
            print('Missing file efficiency')
    axarr[1].plot(phase_array, eff_array / (-psi_2 / psi_1), 'o', color='C1', label='$2$', markersize=8)

    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1].set_ylim((0, 1.1))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].yaxis.offsetText.set_fontsize(14)
    axarr[1].tick_params(axis='both', labelsize=14)
    axarr[1].set_yticks([0, 0.5, 1.0])
    axarr[1].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
    axarr[1].set_xticklabels(['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    leg = axarr[1].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.55, 0.07, r'$n \phi\ (\rm rad)$', fontsize=20, ha='center')
    f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
    f.text(0.03, 0.37, r'$\mathbf{b)}$', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_power_phi_single(target_dir):  # plot of the power as a function of the phase offset
    colorlst = ['C2', 'C3', 'C1', 'C4']
    markerlst = ['D', 's', 'o', 'v']
    Ecouple_array = array([2.0, 8.0, 16.0, 32.0])

    output_file_name = (target_dir
                        + "Power_ATP_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.axhline(0, color='black', linewidth=1)

    # Fokker-Planck results (barriers)
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = (target_dir + "plotting_data/" + "flux_power_efficiency_"
                           + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            phase_array = data_array[:, 0]
            power_y_array = data_array[:, 1]

            ax.plot(phase_array, -2*pi*power_y_array*timescale, linestyle='-', marker=markerlst[ii],
                    label=f'${Ecouple}$', markersize=8, linewidth=2, color=colorlst[ii])
        except OSError:
            print('Missing file')

    # Infinite coupling result
    input_file_name = (target_dir + "plotting_data/"
                       + "Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
    data_array = loadtxt(input_file_name, usecols=(0, 1))
    phase_array = data_array[:, 0]
    power_y_array = -psi_2 * data_array[:, 1]

    ax.plot(phase_array[:61], 2*pi*power_y_array[:61]*timescale, '-', label=f'$\infty$', linewidth=2, color='C6')
    ax.tick_params(axis='both', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.set_xlim((0, 2.1))
    ax.set_ylim((-41, 31))
    ax.set_yticks([-40, -20, 0, 20])

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], title=r'$\beta E_{\rm couple}$', fontsize=14, loc=[0.8, 0.08],
                    frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.55, 0.02, r'$n \phi\ (\rm rad)$', fontsize=20, ha='center')
    plt.ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    plt.xticks([0, pi / 9, 2 * pi / 9, pi / 3, 4 * pi / 9, 5 * pi / 9, 2 * pi / 3],
               ['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    f.tight_layout()
    f.subplots_adjust(bottom=0.14)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_nn_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = array([4.0, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    output_file_name = (
            target_dir + "power_nn_efficiency_Ecouple_plot_more_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        power_y_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                    usecols=(0, 4))
                if len(data_array) == 2:  # data format varies a little
                    power_y = array(data_array[1])
                else:
                    power_y = array(data_array[0, 1])
                power_y_array = append(power_y_array, power_y)
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))

        # Infinite coupling data
        input_file_name = (target_dir + "plotting_data/" +
        "Power_ATP_Ecouple_inf_no_n1_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat")
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')

        axarr[0, 0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])
        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])

    axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0, 0].yaxis.offsetText.set_fontsize(14)
    axarr[0, 0].tick_params(axis='y', labelsize=14)
    axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr[0, 0].spines['right'].set_visible(False)
    axarr[0, 0].spines['top'].set_visible(False)
    axarr[0, 0].set_ylim((0, 20))
    axarr[0, 0].set_xlim((7, None))
    axarr[0, 0].set_yticks([0, 5, 10, 15, 20])

    axarr[0, 1].spines['right'].set_visible(False)
    axarr[0, 1].spines['top'].set_visible(False)
    axarr[0, 1].spines['left'].set_visible(False)
    axarr[0, 1].set_xticks([300])
    axarr[0, 1].set_xticklabels(['$\infty$'])
    axarr[0, 1].tick_params(axis='y', color='white')

    # broken axis
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=axarr[0, 0].transAxes, color='k', clip_on=False)
    axarr[0, 0].plot((1 - 0.3 * d, 1 + 0.3 * d), (-d, +d), **kwargs)
    kwargs.update(transform=axarr[0, 1].transAxes)  # switch to the bottom axes
    axarr[0, 1].plot((-2.5 * d - 0.05, +2.5 * d - 0.05), (-d, +d), **kwargs)

    #########################################################
    # efficiency plot
    # axarr[1, 0].axhline(0, color='black', linewidth=1, label='_nolegend_')
    axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')
    axarr[1, 1].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=5)
                if data_array.size == 1:  # data format varies a little
                    eff_array = append(eff_array, data_array)
                else:
                    eff_array = append(eff_array, data_array[0])
            except OSError:
                print('Missing file efficiency')

        # infinite coupling value
        axarr[1, 0].plot(Ecouple_array_tot, eff_array/0.5, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])
        axarr[1, 1].plot([300], 1, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])

    axarr[1, 0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr[1, 0].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1, 0].set_xscale('log')
    axarr[1, 0].set_xlim((7, None))
    axarr[1, 0].set_ylim((0, None))
    axarr[1, 0].spines['right'].set_visible(False)
    axarr[1, 0].spines['top'].set_visible(False)
    axarr[1, 0].set_yticks([0, 0.5, 1.0])
    axarr[1, 0].tick_params(axis='both', labelsize=14)
    axarr[1, 0].set_xticks([10, 100])
    axarr[1, 0].set_xticklabels(['$10^1$', '$10^2$'])

    axarr[1, 1].spines['right'].set_visible(False)
    axarr[1, 1].spines['top'].set_visible(False)
    axarr[1, 1].spines['left'].set_visible(False)
    axarr[1, 1].set_xticks([300])
    axarr[1, 1].set_xticklabels(['$\infty$'])
    axarr[1, 1].set_xlim((295, 305))
    axarr[1, 1].tick_params(axis='y', color='white')
    axarr[1, 1].tick_params(axis='x', labelsize=14)

    # broken axis
    kwargs = dict(transform=axarr[1, 0].transAxes, color='k', clip_on=False)
    axarr[1, 0].plot((1 - 0.3 * d, 1 + 0.3 * d), (-2 * d, +2 * d), **kwargs)
    kwargs.update(transform=axarr[1, 1].transAxes)  # switch to the bottom axes
    axarr[1, 1].plot((-2.5 * d - 0.05, +2.5 * d - 0.05), (-2 * d, +2 * d), **kwargs)

    leg = axarr[1, 0].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o} = n_1$', fontsize=14,
                             loc='lower right', frameon=False, ncol=3)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.05, 0.92, r'$\mathbf{a)}$', ha='center', fontsize=20)
    f.text(0.05, 0.37, r'$\mathbf{b)}$', ha='center', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_nn_power_efficiency_phi(target_dir):  # plot power and efficiency as a function of the coupling strength
    phase_array = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319])
    Ecouple_array = array([16.0])
    n_labels = ['$1$', '$2$', '$3$', '$6$', '$12$']
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']

    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 4.5))

    output_file_name = (
            target_dir + "power_efficiency_phi_vary_n_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis

    # zero-barrier results
    input_file_name = (
                target_dir + "plotting_data/"
                + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].axhline(2*pi*power_y[28]*timescale, color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    for i, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate(Ecouple_array):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                    usecols=4)
                if num_min == 3.0:
                    power_y = array(data_array[::2])
                else:
                    power_y = array(data_array)
            except OSError:
                print('Missing file flux')
        if num_min != 3.0:
            power_y = append(power_y, power_y[0])
        axarr[0].plot(phase_array, -2*pi*power_y*timescale, '-', markersize=8, color=color_lst[i],
                      marker=markerlst[i], label=n_labels[i])

    axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='both', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].set_ylim((0, None))
    axarr[0].set_xlim((0, 6.3))

    leg = axarr[0].legend(title=r'$n_{\rm o} = n_1$', fontsize=14, loc='lower center', frameon=False, ncol=3)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    #########################################################
    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x-axis
    axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')  # max efficiency
    axarr[1].set_aspect(1.5)

    # Fokker-Planck results (barriers)
    for i, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate(Ecouple_array):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=5)
                if num_min == 3.0:
                    eff_array = array(data_array[::2])
                else:
                    eff_array = array(data_array)
            except OSError:
                print('Missing file efficiency')
        if num_min != 3.0:
            eff_array = append(eff_array, eff_array[0])
        axarr[1].plot(phase_array, eff_array / (-psi_2 / psi_1), marker=markerlst[i], label=n_labels[i],
                      markersize=8, color=color_lst[i])

    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1].set_ylim((0, 1.1))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].yaxis.offsetText.set_fontsize(14)
    axarr[1].tick_params(axis='both', labelsize=14)
    axarr[1].set_yticks([0, 0.5, 1.0])
    axarr[1].set_xticks([0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi])
    axarr[1].set_xticklabels(['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    f.text(0.55, 0.01, r'$n \phi \ (\rm rad)$', fontsize=20, ha='center')  # xlabel seems broken
    f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
    f.text(0.03, 0.4, r'$\mathbf{b)}$', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_n0_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = array(
        [8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    output_file_name = (
                target_dir + "/power_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    # power plot
    # Fokker-Planck results (barriers
    for j, num_min in enumerate(min_array):
        power_y_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple),
                    usecols=4)
                power_y = array(data_array)
                if power_y.size == 1:  # data format varies a little
                    power_y_array = append(power_y_array, power_y)
                else:
                    power_y_array = append(power_y_array, power_y[0])
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple))

        axarr[0, 0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])

        # infinite coupling result
        input_file_name = (target_dir + "plotting_data/" +
                           "Power_ATP_Ecouple_inf_no_varies_n1_3.0_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat"
                           )
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')

        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, color=color_lst[j])

    axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0, 0].yaxis.offsetText.set_fontsize(14)
    axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr[0, 0].spines['right'].set_visible(False)
    axarr[0, 0].spines['top'].set_visible(False)
    axarr[0, 0].set_ylim((0, None))
    axarr[0, 0].set_xlim((7.5, None))
    axarr[0, 0].tick_params(axis='both', labelsize=14)
    axarr[0, 0].set_yticks([0, 5, 10, 15, 20])

    axarr[0, 1].spines['right'].set_visible(False)
    axarr[0, 1].spines['top'].set_visible(False)
    axarr[0, 1].spines['left'].set_visible(False)
    axarr[0, 1].set_xticks([300])
    axarr[0, 1].set_xticklabels(['$\infty$'])
    axarr[0, 1].tick_params(axis='y', color='white')

    # broken axes
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=axarr[0, 0].transAxes, color='k', clip_on=False)
    axarr[0, 0].plot((1 - 0.3*d, 1 + 0.3*d), (-d, +d), **kwargs)
    kwargs.update(transform=axarr[0, 1].transAxes)  # switch to the bottom axes
    axarr[0, 1].plot((-2.5*d-0.05, +2.5*d-0.05), (-d, +d), **kwargs)

    #########################################################
    # efficiency plot
    axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')  # max efficiency
    axarr[1, 1].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple), usecols=5)
                if data_array.size == 1:
                    eff_array = append(eff_array, data_array)
                else:
                    eff_array = append(eff_array, data_array[0])
            except OSError:
                print('Missing file efficiency')

        axarr[1, 0].plot(Ecouple_array_tot, eff_array / (-psi_2 / psi_1), marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])
        axarr[1, 1].plot([300], [1], marker=markerlst[j], markersize=6, color=color_lst[j])  # infinite coupling

    axarr[1, 0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr[1, 0].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1, 0].set_xscale('log')
    axarr[1, 0].set_ylim((0, None))
    axarr[1, 0].spines['right'].set_visible(False)
    axarr[1, 0].spines['top'].set_visible(False)
    axarr[1, 0].set_yticks([0, 0.5, 1.0])
    axarr[1, 0].tick_params(axis='both', labelsize=14)
    axarr[1, 0].set_xticks([10, 100])
    axarr[1, 0].set_xticklabels(['$10^1$', '$10^2$'])

    axarr[1, 1].spines['right'].set_visible(False)
    axarr[1, 1].spines['top'].set_visible(False)
    axarr[1, 1].spines['left'].set_visible(False)
    axarr[1, 1].set_xticks([300])
    axarr[1, 1].set_xticklabels(['$\infty$'])
    axarr[1, 1].set_xlim((295, 305))
    axarr[1, 1].tick_params(axis='y', color='white')
    axarr[1, 1].tick_params(axis='x', labelsize=14)

    # broken axes
    kwargs = dict(transform=axarr[1, 0].transAxes, color='k', clip_on=False)
    axarr[1, 0].plot((1 - 0.3*d, 1 + 0.3*d), (-2*d, +2*d), **kwargs)
    kwargs.update(transform=axarr[1, 1].transAxes)  # switch to the bottom axes
    axarr[1, 1].plot((-2.5*d-0.05, +2.5*d-0.05), (-2*d, +2*d), **kwargs)

    leg = axarr[1, 0].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o}$', fontsize=14,
                             loc='lower center', frameon=False, ncol=3)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.67, 0.25, r'$n_1=3$', ha='center', fontsize=14)
    f.text(0.05, 0.92, r'$\mathbf{a)}$', ha='center', fontsize=20)
    f.text(0.05, 0.37, r'$\mathbf{b)}$', ha='center', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, 3.0))

if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/" + \
                 "prediction/fokker_planck/working_directory_cython/"
    flux_power_efficiency(target_dir)
    # plot_power_efficiency_Ecouple(target_dir)
    # plot_power_Ecouple_grid(target_dir)
    # plot_power_efficiency_phi(target_dir)
    # plot_power_phi_single(target_dir)
    # plot_nn_power_efficiency_Ecouple(target_dir)
    # plot_nn_power_efficiency_phi(target_dir)
    # plot_n0_power_efficiency_Ecouple(target_dir)