from numpy import array, linspace, loadtxt, append, pi, empty, sqrt, zeros, asarray, trapz, exp, argmax, max, min
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
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

Ecouple_array_tot = array([2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 22.63, 24.0,
                           32.0, 45.25, 64.0, 90.51, 128.0])  # coupling
min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])  # number of energy minima/ barriers
barrier_heights = array([2.0, 4.0])  # barrier heights
psi1_array = array([2.0, 4.0, 8.0])  # driving force on Fo
psi2_array = array([-0.25, -0.5, -1.0, -2.0, -4.0])  # driving force on F1
psi_ratio = array([8, 4, 2])  # ratio of driving forces

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


def flux_power_efficiency(raw_data_dir, input_dir): # processing of raw data
    phase_array = array([0.0])
    flag = 0  # flag to make sure data only gets saved to file if processing was successful

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            for Ecouple in Ecouple_array_tot:
                integrate_flux_X = empty(phase_array.size)
                integrate_flux_Y = empty(phase_array.size)
                integrate_power_X = empty(phase_array.size)
                integrate_power_Y = empty(phase_array.size)

                for ii, phase_shift in enumerate(phase_array):
                    input_file_name = (raw_data_dir + "200506_4kTbarrier/spectral/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")

                    output_file_name = (input_dir + "Test/" + "flux_power_efficiency_" +
                                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")

                    print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, ",
                          f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")

                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1,
                                                                    num_minima2, phase_shift),
                                             usecols=(0, 3, 4, 5, 6, 7, 8))
                        N = int(sqrt(len(data_array)))  # check grid size
                        print('Grid size: ', N)
                        dx = 2 * math.pi / N

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

                        flag = 0

                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                     phase_shift))
                        flag = 1

                if flag == 0:
                    if abs(psi_1) <= abs(psi_2):
                        efficiency_ratio = -(integrate_power_X/integrate_power_Y)
                    else:
                        efficiency_ratio = -(integrate_power_Y/integrate_power_X)

                    with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), "w") \
                            as ofile:
                        for ii, phase_shift in enumerate(phase_array):
                            ofile.write(
                                f"{phase_shift:.15e}" + "\t"
                                + f"{integrate_flux_X[ii]:.15e}" + "\t"
                                + f"{integrate_flux_Y[ii]:.15e}" + "\t"
                                + f"{integrate_power_X[ii]:.15e}" + "\t"
                                + f"{integrate_power_Y[ii]:.15e}" + "\t"
                                + f"{efficiency_ratio[ii]:.15e}" + "\n")
                        ofile.flush()


def plot_power_efficiency_Ecouple(input_dir, output_dir):  # plot power and efficiency vs coupling strength
    barrier_label = ['$2$', '$4$']
    colorlst = ['C1', 'C9']
    offset = [0, 4]

    output_file_name = (output_dir + "P_ATP_eff_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_.pdf")

    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 8))

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis
    maxpower = 2 * pi * 0.000085247 * timescale
    axarr[0].axhline(maxpower, color='C1', linestyle=':', linewidth=2)  # line at infinite power coupling
    axarr[0].fill_between([1, 250], 0, 31, facecolor='grey', alpha=0.2)  # shading power output

    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x axis
    axarr[1].axhline(1, color='black', linestyle=':', linewidth=2)  # max efficiency
    axarr[1].fill_between([1, 250], 0, 1, facecolor='grey', alpha=0.2)  # shading power output

    # no-barrier results
    input_file_name = (input_dir + "Driving_forces/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    Ecouple_array2 = array(data_array[:, 0])
    flux_x_array = array(data_array[:, 1])
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', color='C0', label='$0$', linewidth=2)
    axarr[1].plot(Ecouple_array2, flux_y_array / flux_x_array, '-', color='C0', linewidth=2)

    # peak position estimate output power from theory
    Ecouple_est = 3.31 + 4 * pi * (psi_1 - psi_2) / 9
    axarr[0].axvline(Ecouple_est, color='black', linestyle='-', linewidth=2)

    # Fokker-Planck barrier results
    for j, E0 in enumerate(barrier_heights):
        E1 = E0
        power_y_array = []
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (input_dir + "Driving_forces/" + "flux_power_efficiency_"
                               + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                    usecols=(4, 5))
                power_y_array = append(power_y_array, data_array[0])
                eff_array = append(eff_array, data_array[1])
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(4.0, 4.0, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
        # dashed vertical lines at max power
        axarr[0].axvline(Ecouple_array_tot[argmax(-power_y_array)], linestyle=(offset[j], (4, 4)), color=colorlst[j],
                         linewidth=2)
        axarr[1].axvline(Ecouple_array_tot[argmax(-power_y_array)], linestyle=(offset[j], (4, 4)), color=colorlst[j],
                         linewidth=2)
        # power plot
        axarr[0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, 'o', color=colorlst[j], label=barrier_label[j],
                      markersize=8)
        # scaled efficiency plot
        axarr[1].plot(Ecouple_array_tot, eff_array / (-psi_2 / psi_1), 'o', color=colorlst[j], markersize=8)

    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].spines['bottom'].set_visible(False)
    axarr[0].set_xlim((1.7, 135))
    axarr[0].set_ylim((None, 31))

    leg = axarr[0].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

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


def plot_power_Ecouple_grid(input_dir, output_dir):  # grid of plots of the flux as a function of the phase offset
    colorlst = ['C1', 'C9']
    offset = [0, 2]

    output_file_name = (output_dir + "P_ATP_Ecouple_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")

    f, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(8, 6))

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print('Chemical driving forces:', psi_1, psi_2)

            # zero-barrier result
            input_file_name = (input_dir + "Driving_forces/" + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2

            axarr[i, j].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', color='C0', linewidth=3)

            # peak position estimate from theory
            Ecouple_est = 3.31 + 4 * pi * (psi_1 - psi_2) / 9
            axarr[i, j].axvline(Ecouple_est, color='black', linestyle='-', linewidth=3)

            # Fokker-Planck results
            for k, E0 in enumerate(barrier_heights):
                E1 = E0
                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array_tot):
                    input_file_name = (input_dir + "Driving_forces/" + "flux_power_efficiency_" +
                                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                            usecols=4)
                        power_y_array = append(power_y_array, data_array)
                    except OSError:
                        print('Missing file power')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                peak = argmax(-power_y_array)
                # vertical dashed line at max power
                axarr[i, j].axvline(Ecouple_array_tot[peak], linestyle=(offset[k], (2, 2)), color=colorlst[k],
                                    linewidth=3)
                # power plot
                axarr[i, j].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, '.', color=colorlst[k],
                                 markersize=14)

            axarr[i, j].set_xscale('log')
            axarr[i, j].set_yscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((2, 150))
            elif j == 1:
                axarr[i, j].set_xlim((3, 150))
            else:
                axarr[i, j].set_xlim((4, 150))

            if i == 0:
                axarr[i, j].set_ylim((3*10**(-2), 10))
                axarr[i, j].set_yticks([0.1, 1, 10])
            elif i == 1:
                axarr[i, j].set_ylim((2*10**(-1), 40))
                axarr[i, j].set_yticks([1, 10])
            else:
                axarr[i, j].set_ylim((2, 130))
                axarr[i, j].set_yticks([10, 100])

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
                axarr[i, j].tick_params(axis='x', colors='red', which='major')
                axarr[i, j].tick_params(axis='y', colors='red', which='major')
                axarr[i, j].spines['left'].set_color('red')
                axarr[i, j].spines['bottom'].set_color('red')
            else:
                axarr[i, j].tick_params(axis='both', labelsize=18)
            axarr[i, j].tick_params(axis='x', colors='white', which='minor')
            axarr[i, j].tick_params(axis='y', colors='white', which='minor')

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=24)
    f.text(0.01, 0.5, r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1})$', va='center', rotation='vertical', fontsize=24)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=24)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (k_{\rm B} T / \rm rad)$', va='center', rotation=270, fontsize=24)

    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_power_efficiency_phi(input_dir, output_dir): # plot power and efficiency as a function of the coupling strength
    output_file_name = (output_dir + "P_ATP_efficiency_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 6))
    colorlst = ['C1', 'C9']
    labels = ['$2$', '$4$']

    # zero-barrier results
    input_file_name = (input_dir + "Driving_forces/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].axhline(2 * pi * power_y[28] * timescale, color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    for k, E0 in enumerate([2.0, 4.0]):
        E1 = E0
        for ii, Ecouple in enumerate([16.0]):
            input_file_name = (input_dir + "Phase_offset/" + "flux_power_efficiency_" +
                               "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                    usecols=(0, 4, 5))
                phase_array = array(data_array[:, 0])
                power_y = array(data_array[:, 1])
                eff_array = array(data_array[:, 2])
            except OSError:
                print('Missing file flux')
        # plot power
        axarr[0].plot(phase_array, -2*pi*power_y*timescale, 'o', color=colorlst[k], label=labels[k], markersize=8)
        # plot efficiency
        axarr[1].plot(phase_array, eff_array / (-psi_2 / psi_1), 'o', color=colorlst[k], label=labels[k], markersize=8)

    axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='both', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].set_ylim((0, 31))
    axarr[0].set_xlim((0, 2.1))
    axarr[0].set_yticks([0, 10, 20, 30])

    axarr[1].axhline(0, color='black', linewidth=1)  # x-axis
    axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')  # max efficiency
    axarr[1].set_aspect(0.5)
    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1].set_ylim((0, 1.1))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].yaxis.offsetText.set_fontsize(14)
    axarr[1].tick_params(axis='both', labelsize=14)
    axarr[1].set_yticks([0, 0.5, 1.0])
    axarr[1].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
    axarr[1].set_xticklabels(['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    leg = axarr[1].legend([Line2D([0], [0], color='C0', lw=2), Line2D([0], [0], marker='o', color=colorlst[0], lw=0),
                           Line2D([0], [0], marker='o', color=colorlst[1], lw=0)], ['$0$', '$2$', '$4$'], ncol=3,
                          title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.55, 0.07, r'$n \phi\ (\rm rad)$', fontsize=20, ha='center')
    f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
    f.text(0.03, 0.37, r'$\mathbf{b)}$', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, 16.0))


def plot_power_phi_single(input_dir, output_dir):  # plot of the power as a function of the phase offset
    colorlst = ['C7', 'C3', 'C1', 'C4']
    markerlst = ['D', 's', 'o', 'v']
    Ecouple_array = array([2.0, 8.0, 16.0, 32.0])

    output_file_name = (output_dir + "P_ATP_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.axhline(0, color='black', linewidth=1)

    # Fokker-Planck results (barriers)
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = (input_dir + "Phase_offset/" + "flux_power_efficiency_"
                           + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            phase_array = data_array[:, 0]
            power_y_array = data_array[:, 1]

            ax.plot(phase_array, -2 * pi * power_y_array * timescale, linestyle='-', marker=markerlst[ii],
                    label=f'${int(Ecouple)}$', markersize=8, linewidth=2, color=colorlst[ii])
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

    # Infinite coupling result
    input_file_name = (input_dir + "Phase_offset/" + "Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
    data_array = loadtxt(input_file_name, usecols=(0, 1))
    phase_array = data_array[:, 0]
    power_y_array = -psi_2 * data_array[:, 1]

    ax.plot(phase_array[:61], 2*pi*power_y_array[:61]*timescale, '-', label=r'$\infty$', linewidth=2, color='C6')
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


def plot_efficiency_Ecouple_grid(input_dir, output_dir):
    colorlst = ['C1', 'C9']

    output_file_name = (output_dir + "Eff_Ecouple_theory_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")

    f, axarr = plt.subplots(3, 3, sharex='all', sharey='col', figsize=(8, 6))

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print('Chemical driving forces:', psi_1, psi_2)

            # zero-barrier result
            input_file_name = (input_dir + "Driving_forces/" + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_x_array = array(data_array[:, 1])
            flux_y_array = array(data_array[:, 2])
            eff_array = flux_y_array / flux_x_array
            axarr[i, j].plot(Ecouple_array2, eff_array, '-', color='C0', linewidth=3)

            # Fokker-Planck results
            for k, E0 in enumerate(barrier_heights):
                E1 = E0
                eff_array = []
                for ii, Ecouple in enumerate(Ecouple_array_tot):
                    input_file_name = (input_dir + "Driving_forces/" + "flux_power_efficiency_" +
                                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                            usecols=5)
                        eff_array = append(eff_array, data_array)
                    except OSError:
                        print('Missing file efficiency')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                axarr[i, j].plot(Ecouple_array_tot, eff_array / (-psi_2/psi_1), '.', color=colorlst[k],
                                 markersize=14)

            # rate calculations theory line
            pos = linspace(1, 150, 200)  # array of coupling strengths
            theory = 1 - 3 * exp((pi/3) * (psi_1 - psi_2) - 0.75 * pos)
            axarr[i, j].plot(pos, theory, '--', color='black', linewidth=2)

            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].set_xticks([1., 10., 100.])
            axarr[i, j].set_ylim((0, 1.05))
            axarr[i, j].set_xlim((1, 150))

            if j == 0 and i > 0:
                axarr[i, j].yaxis.offsetText.set_fontsize(0)
            else:
                axarr[i, j].yaxis.offsetText.set_fontsize(14)

            if j > 0:
                axarr[i, j].set_yticklabels([])

            if j == psi1_array.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi_ratio[::-1][i], labelpad=16, rotation=270, fontsize=18)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.0f$' % psi1_array[::-1][j], fontsize=18)

            axarr[i, j].tick_params(axis='both', labelsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=24)
    f.text(0.01, 0.5, r'$\eta / \eta^{\rm max}$', va='center', rotation='vertical',
           fontsize=24)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=24)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (k_{\rm B} T / \rm rad)$', va='center', rotation=270, fontsize=24)

    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_nn_power_efficiency_Ecouple(input_dir, output_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = array([4.0, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    output_file_name = (output_dir + "P_ATP_efficiency_Ecouple_nn_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")

    axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')  # max efficiency
    axarr[1, 1].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        power_y_array = []
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if num_min != 3.0:
                input_file_name = (input_dir + "Number_barriers/" + "flux_power_efficiency_" +
                                   "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            else:
                input_file_name = (input_dir + "Driving_forces/" + "flux_power_efficiency_" +
                                   "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                    usecols=(0, 4, 5))
                if len(data_array) == 3:  # data format varies a little
                    power_y = array(data_array[1])
                    eff = array(data_array[2])
                else:
                    power_y = array(data_array[0, 1])
                    eff = array(data_array[0, 2])
                power_y_array = append(power_y_array, power_y)
                eff_array = append(eff_array, eff)
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))

        # Infinite coupling data
        input_file_name = (input_dir + "Number_barriers/" +
                           "Power_ATP_Ecouple_inf_no_n1_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat")
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')
        # plot power
        axarr[0, 0].plot(Ecouple_array_tot, -2 * pi * power_y_array * timescale, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])
        # add point at Ecouple=300
        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])
        # plot efficiency
        axarr[1, 0].plot(Ecouple_array_tot, eff_array/0.5, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])
        # add point at Ecouple=300
        axarr[1, 1].plot([300], 1, marker=markerlst[j], markersize=6, linestyle='-', color=color_lst[j])

    # formatting
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
    axarr[0, 1].set_xticklabels([r'$\infty$'])
    axarr[0, 1].tick_params(axis='y', color='white')

    # broken axis
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=axarr[0, 0].transAxes, color='k', clip_on=False)
    axarr[0, 0].plot((1 - 0.3 * d, 1 + 0.3 * d), (-d, +d), **kwargs)
    kwargs.update(transform=axarr[0, 1].transAxes)  # switch to the bottom axes
    axarr[0, 1].plot((-2.5 * d - 0.05, +2.5 * d - 0.05), (-d, +d), **kwargs)

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


def plot_nn_power_efficiency_phi(input_dir, output_dir):  # plot power and efficiency as a function of the coupling strength
    phase_array = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319])
    n_labels = ['$1$', '$2$', '$3$', '$6$', '$12$']
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']

    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 4.5))

    output_file_name = (output_dir + "P_ATP_efficiency_phi_vary_n_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis

    # zero-barrier results
    input_file_name = (input_dir + "Driving_forces/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].axhline(2*pi*power_y[28]*timescale, color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    for i, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate([16.0]):
            if num_min != 3.0:
                input_file_name = (input_dir + "Number_barriers/" + "flux_power_efficiency_"
                                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            else:
                input_file_name = (input_dir + "Phase_offset/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                    usecols=(4, 5))
                if num_min == 3.0:
                    power_y = array(data_array[::2, 0])
                    eff_array = array(data_array[::2, 1])
                else:
                    power_y = array(data_array[:, 0])
                    power_y = append(power_y, power_y[0])
                    eff_array = array(data_array[:, 1])
                    eff_array = append(eff_array, eff_array[0])
            except OSError:
                print('Missing file power')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))
        # plot power
        axarr[0].plot(phase_array, -2 * pi * power_y * timescale, '-', markersize=8, color=color_lst[i],
                      marker=markerlst[i], label=n_labels[i])
        # plot efficiency
        axarr[1].plot(phase_array, eff_array / (-psi_2 / psi_1), marker=markerlst[i], label=n_labels[i],
                      markersize=8, color=color_lst[i])

    # dashed line at max efficiency
    axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')

    # formatting
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

    axarr[1].set_aspect(1.5)
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


def plot_n0_power_efficiency_Ecouple(input_dir, output_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = array([8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])

    output_file_name = (output_dir + "P_ATP_efficiency_Ecouple_n0_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_.pdf")
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')  # max efficiency
    axarr[1, 1].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')

    # Fokker-Planck results (barriers
    for j, num_min in enumerate(min_array):
        power_y_array = []
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if num_min != 3.0:
                input_file_name = (input_dir + "Number_barriers/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            else:
                input_file_name = (input_dir + "Driving_forces/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple),
                    usecols=(4, 5))
                power_y = array(data_array[0])
                eff = array(data_array[1])
                power_y_array = append(power_y_array, power_y)
                eff_array = append(eff_array, eff)
            except OSError:
                print('Missing file power')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple))

        axarr[0, 0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])
        axarr[1, 0].plot(Ecouple_array_tot, eff_array / (-psi_2 / psi_1), marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])

        # infinite coupling result
        input_file_name = (input_dir + "Number_barriers/" +
                           "Power_ATP_Ecouple_inf_no_varies_n1_3.0_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat")
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')
        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, color=color_lst[j])
        axarr[1, 1].plot([300], [1], marker=markerlst[j], markersize=6, color=color_lst[j])  # infinite coupling

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


def plot_power_barrier_height_rigid_coupling(input_dir, output_dir):
    output_file_name = (output_dir + "P_ATP_E0_" + "n0_{0}_n1_{1}_phi_{2}" + "_.pdf")
    f, axarr = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(8, 6))

    psi1_array = array([8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0])
    psi2_array = array([-4.0, -2.0, -1.0, -2.0, -1.0, -0.5, -1.0, -0.5, -0.25])

    colorlst = ['0.7', '0.4', '0.0']
    linelst = ['solid', 'dotted', 'dashed']

    axarr.axvline(0.0, linestyle='-', color='C0', linewidth=2)
    axarr.axvline(2.0, linestyle='-', color='C1', linewidth=2)
    axarr.axvline(4.0, linestyle='-', color='C9', linewidth=2)

    for i in range(len(psi2_array)):
        psi_1 = psi1_array[i]
        psi_2 = psi2_array[i]
        input_file_name = (input_dir + "Barrier_height/" +
                           "flux_rigid_coupling_psi1_{0}_psi2_{1}_n0_{2}_n1_{3}_phase_{4}_outfile.dat")
        data_array = loadtxt(input_file_name.format(psi_1, psi_2, num_minima1, num_minima2, 0.0))
        barrier_array = data_array[:, 0]
        flux_array = data_array[:, 1]
        power_y = -flux_array * psi_2
        axarr.plot(barrier_array, 2 * pi * power_y * timescale, linewidth=2, color=colorlst[i % 3],
                   linestyle=linelst[int(i/3)])

        axarr.yaxis.offsetText.set_fontsize(16)
        axarr.tick_params(axis='both', labelsize=16)
        axarr.set_xlabel(r'$\beta E^\ddagger$', fontsize=20)
        axarr.set_xticks([0, 2, 4, 6, 8, 10])
        axarr.set_xticklabels(['$0$', '$4$', '$8$', '$12$', '$16$', '$20$'])
        axarr.set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
        axarr.spines['right'].set_visible(False)
        axarr.spines['top'].set_visible(False)
        axarr.set_yscale('log')

    leg = axarr.legend([Line2D([0], [0], color=colorlst[2], lw=2), Line2D([0], [0], color=colorlst[1], lw=2),
                        Line2D([0], [0], color=colorlst[0], lw=2), Line2D([0], [0], color=colorlst[2], lw=2),
                        Line2D([0], [0], color=colorlst[2], lw=2, linestyle='dotted'),
                        Line2D([0], [0], color=colorlst[2], lw=2, linestyle='dashed')],
                       ['$8$', '$4$', '$2$', '$8$', '$4$', '$2$'],
                       title=r'$-\mu_{\rm H^+}/\mu_{\rm ATP}, \beta \mu_{\rm H^+} (\rm rad^{-1})$',
                       fontsize=16, loc='upper right', frameon=False, ncol=2)
    leg_title = leg.get_title()
    leg_title.set_fontsize(16)
    leg2 = axarr.legend([Line2D([0], [0], color='C0', lw=2),
                         Line2D([0], [0], color='C1', lw=2),
                         Line2D([0], [0], color='C9', lw=2)],
                        ['$0$', '$2$', '$4$'],
                        title=r'$\beta E_{\rm o} = \beta E_1$',
                        fontsize=16, loc=(0.05, 0.02), frameon=False, ncol=1)
    leg_title2 = leg2.get_title()
    leg_title2.set_fontsize(16)
    axarr.add_artist(leg2)
    axarr.add_artist(leg)

    f.tight_layout()
    f.savefig(output_file_name.format(num_minima1, num_minima2, 0.0))


if __name__ == "__main__":
    raw_data_dir = "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/"
    input_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/"
    output_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/results/FP_Full_2D/"

    #  data preprocessing
    # flux_power_efficiency(raw_data_dir, input_dir)

    #  figures main text
    plot_power_efficiency_Ecouple(input_dir, output_dir)
    plot_power_Ecouple_grid(input_dir, output_dir)
    plot_power_efficiency_phi(input_dir, output_dir)
    plot_power_phi_single(input_dir, output_dir)

    # #  figures SI
    plot_efficiency_Ecouple_grid(input_dir, output_dir)
    plot_nn_power_efficiency_Ecouple(input_dir, output_dir)
    plot_nn_power_efficiency_phi(input_dir, output_dir)
    plot_n0_power_efficiency_Ecouple(input_dir, output_dir)
    plot_power_barrier_height_rigid_coupling(input_dir, output_dir)

