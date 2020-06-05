import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi, meshgrid, shape, amax, amin, zeros, round, append, exp, \
    log, ones, sqrt, floor, shape, diff, sign, nonzero, where
import math
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

N = 360
dx = 2 * math.pi / N
positions = linspace(0, 2 * math.pi - dx, N)
E0 = 2.0
E1 = 2.0
num_minima1 = 10.0
num_minima2 = 3.0

min_array = array([3.0])
# psi1_array = array([1., 2., 4.])
# psi2_array = array([-1., -2.0, -4.])
psi1_array = array([10.0])
psi2_array = array([-9.0])
# psi1_array = array([2.0, 4.0, 8.0])
# psi2_array = array([-0.25, -0.5, -1.0, -2.0,-4.0])
# psi_ratio = array([8, 4, 2])
# Ecouple_array = array([2.0, 8.0, 16.0, 32.0])
# Ecouple_array = array([2.0])
Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])  # twopisweep
# Ecouple_array = array([2.0, 8.0, 16.0, 32.0])
Ecouple_array_extra = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0])  # extra measurements
Ecouple_array_extra2 = array([1.41, 2.83, 5.66, 11.31, 22.63, 45.25, 90.51])
Ecouple_array_tot = array([1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
Ecouple_array_tot = array([1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 22.63,
                           24.0, 32.0, 45.25, 64.0, 90.51, 128.0])
# phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346, 2.79253, 3.14159, 3.49066, 3.83972, 4.18879, 4.53786, 4.88692, 5.23599, 5.58505, 5.93412, 6.28319]) #twopisweep
# phase_array = array([0.0, 0.175, 0.349066, 0.524, 0.698132, 0.873, 1.0472, 1.222, 1.39626, 1.571, 1.74533, 1.92, 2.0944]) #selection of twopisweep
# phase_array_test = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944])
phase_array = array([0.0])

phi_array = linspace(0, 2 * pi, N)

colorlist = linspace(0, 1, len(phase_array))
colorlst = ['C2', 'C3', 'C1', 'C4']
# label_lst=['0', '$\pi/9$', '$2\pi/9$', '$\pi/3$', '$4 \pi/9$', '$5 \pi/9$']
# label_lst=['$0$', ' ', ' ', '$\pi/3$', ' ', ' ', '$2 \pi/3$']
label_lst = ['$0$', ' ', ' ', '$1/6$', ' ', ' ', '$1/3$']
size_lst = [8, 7, 6, 5, 4, 3]
marker_lst = [".", "v", "+", "s", "p", "^", "x", "D", ".", "v", "+", "s", "p", "^", "x", "D"]
ticklst = linspace(0, 2 * math.pi, 7)
# ylabels_flux = [-0.0003, -0.0002, -0.0001, 0]
ylabels_flux = [-0.0006, -0.0003, 0, 0.0003]
ylabels_eff = [0, 0.5, 1.0]
# ticklabels=['0', '$\pi/6$', '$\pi/3$', '$\pi/2$', '$2 \pi / 3$']
ticklabels = ['$0$', '', '$2\pi/3$', '', '$4\pi/3$', '', '$2 \pi$']


def calc_flux(p_now, drift_at_pos, diffusion_at_pos, flux_array, N):
    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] = (
            (drift_at_pos[0, 0, 0] * p_now[0, 0])
            - (diffusion_at_pos[0, 1, 0] * p_now[1, 0] - diffusion_at_pos[0, N - 1, 0] * p_now[N - 1, 0]) / (2.0 * dx)
            - (diffusion_at_pos[1, 0, 1] * p_now[0, 1] - diffusion_at_pos[1, 0, N - 1] * p_now[0, N - 1]) / (2.0 * dx)
    )
    flux_array[0, 0, N - 1] = (
            (drift_at_pos[0, 0, N - 1] * p_now[0, N - 1])
            - (diffusion_at_pos[0, 1, N - 1] * p_now[1, N - 1] - diffusion_at_pos[0, N - 1, N - 1] * p_now[
        N - 1, N - 1]) / (2.0 * dx)
            - (diffusion_at_pos[1, 0, 0] * p_now[0, 0] - diffusion_at_pos[1, 0, N - 2] * p_now[0, N - 2]) / (2.0 * dx)
    )
    flux_array[0, N - 1, 0] = (
            (drift_at_pos[0, N - 1, 0] * p_now[N - 1, 0])
            - (diffusion_at_pos[0, 0, 0] * p_now[0, 0] - diffusion_at_pos[0, N - 2, 0] * p_now[N - 2, 0]) / (2.0 * dx)
            - (diffusion_at_pos[1, N - 1, 1] * p_now[N - 1, 1] - diffusion_at_pos[1, N - 1, N - 1] * p_now[
        N - 1, N - 1]) / (2.0 * dx)
    )
    flux_array[0, N - 1, N - 1] = (
            (drift_at_pos[0, N - 1, N - 1] * p_now[N - 1, N - 1])
            - (diffusion_at_pos[0, 0, N - 1] * p_now[0, N - 1] - diffusion_at_pos[0, N - 2, N - 1] * p_now[
        N - 2, N - 1]) / (2.0 * dx)
            - (diffusion_at_pos[1, N - 1, 0] * p_now[N - 1, 0] - diffusion_at_pos[1, N - 1, N - 2] * p_now[
        N - 1, N - 2]) / (2.0 * dx)
    )

    # second component
    flux_array[1, 0, 0] = (
            (drift_at_pos[1, 0, 0] * p_now[0, 0])
            - (diffusion_at_pos[2, 1, 0] * p_now[1, 0] - diffusion_at_pos[2, N - 1, 0] * p_now[N - 1, 0]) / (2.0 * dx)
            - (diffusion_at_pos[3, 0, 1] * p_now[0, 1] - diffusion_at_pos[3, 0, N - 1] * p_now[0, N - 1]) / (2.0 * dx)
    )
    flux_array[1, 0, N - 1] = (
            (drift_at_pos[1, 0, N - 1] * p_now[0, N - 1])
            - (diffusion_at_pos[2, 1, N - 1] * p_now[1, N - 1] - diffusion_at_pos[2, N - 1, N - 1] * p_now[
        N - 1, N - 1]) / (2.0 * dx)
            - (diffusion_at_pos[3, 0, 0] * p_now[0, 0] - diffusion_at_pos[3, 0, N - 2] * p_now[0, N - 2]) / (2.0 * dx)
    )
    flux_array[1, N - 1, 0] = (
            (drift_at_pos[1, N - 1, 0] * p_now[N - 1, 0])
            - (diffusion_at_pos[2, 0, 0] * p_now[0, 0] - diffusion_at_pos[2, N - 2, 0] * p_now[N - 2, 0]) / (2.0 * dx)
            - (diffusion_at_pos[3, N - 1, 1] * p_now[N - 1, 1] - diffusion_at_pos[3, N - 1, N - 1] * p_now[
        N - 1, N - 1]) / (2.0 * dx)
    )
    flux_array[1, N - 1, N - 1] = (
            (drift_at_pos[1, N - 1, N - 1] * p_now[N - 1, N - 1])
            - (diffusion_at_pos[2, 0, N - 1] * p_now[0, N - 1] - diffusion_at_pos[2, N - 2, N - 1] * p_now[
        N - 2, N - 1]) / (2.0 * dx)
            - (diffusion_at_pos[3, N - 1, 0] * p_now[N - 1, 0] - diffusion_at_pos[3, N - 1, N - 2] * p_now[
        N - 1, N - 2]) / (2.0 * dx)
    )

    for i in range(1, N - 1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] = (
                (drift_at_pos[0, 0, i] * p_now[0, i])
                - (diffusion_at_pos[0, 1, i] * p_now[1, i] - diffusion_at_pos[0, N - 1, i] * p_now[N - 1, i]) / (
                            2.0 * dx)
                - (diffusion_at_pos[1, 0, i + 1] * p_now[0, i + 1] - diffusion_at_pos[1, 0, i - 1] * p_now[
            0, i - 1]) / (2.0 * dx)
        )
        flux_array[0, i, 0] = (
                (drift_at_pos[0, i, 0] * p_now[i, 0])
                - (diffusion_at_pos[0, i + 1, 0] * p_now[i + 1, 0] - diffusion_at_pos[0, i - 1, 0] * p_now[
            i - 1, 0]) / (2.0 * dx)
                - (diffusion_at_pos[1, i, 1] * p_now[i, 1] - diffusion_at_pos[1, i, N - 1] * p_now[i, N - 1]) / (
                            2.0 * dx)
        )
        flux_array[0, N - 1, i] = (
                (drift_at_pos[0, N - 1, i] * p_now[N - 1, i])
                - (diffusion_at_pos[0, 0, i] * p_now[0, i] - diffusion_at_pos[0, N - 2, i] * p_now[N - 2, i]) / (
                            2.0 * dx)
                - (diffusion_at_pos[1, N - 1, i + 1] * p_now[N - 1, i + 1] - diffusion_at_pos[1, N - 1, i - 1] * p_now[
            N - 1, i - 1]) / (2.0 * dx)
        )
        flux_array[0, i, N - 1] = (
                (drift_at_pos[0, i, N - 1] * p_now[i, N - 1])
                - (diffusion_at_pos[0, i + 1, N - 1] * p_now[i + 1, N - 1] - diffusion_at_pos[0, i - 1, N - 1] * p_now[
            i - 1, N - 1]) / (2.0 * dx)
                - (diffusion_at_pos[1, i, 0] * p_now[i, 0] - diffusion_at_pos[1, i, N - 2] * p_now[i, N - 2]) / (
                            2.0 * dx)
        )

        # second component
        flux_array[1, 0, i] = (
                (drift_at_pos[1, 0, i] * p_now[0, i])
                - (diffusion_at_pos[2, 1, i] * p_now[1, i] - diffusion_at_pos[2, N - 1, i] * p_now[N - 1, i]) / (
                            2.0 * dx)
                - (diffusion_at_pos[3, 0, i + 1] * p_now[0, i + 1] - diffusion_at_pos[3, 0, i - 1] * p_now[
            0, i - 1]) / (2.0 * dx)
        )
        flux_array[1, i, 0] = (
                (drift_at_pos[1, i, 0] * p_now[i, 0])
                - (diffusion_at_pos[2, i + 1, 0] * p_now[i + 1, 0] - diffusion_at_pos[2, i - 1, 0] * p_now[
            i - 1, 0]) / (2.0 * dx)
                - (diffusion_at_pos[3, i, 1] * p_now[i, 1] - diffusion_at_pos[3, i, N - 1] * p_now[i, N - 1]) / (
                            2.0 * dx)
        )
        flux_array[1, N - 1, i] = (
                (drift_at_pos[1, N - 1, i] * p_now[N - 1, i])
                - (diffusion_at_pos[2, 0, i] * p_now[0, i] - diffusion_at_pos[2, N - 2, i] * p_now[N - 2, i]) / (
                            2.0 * dx)
                - (diffusion_at_pos[3, N - 1, i + 1] * p_now[N - 1, i + 1] - diffusion_at_pos[3, N - 1, i - 1] * p_now[
            N - 1, i - 1]) / (2.0 * dx)
        )
        flux_array[1, i, N - 1] = (
                (drift_at_pos[1, i, N - 1] * p_now[i, N - 1])
                - (diffusion_at_pos[2, i + 1, N - 1] * p_now[i + 1, N - 1] - diffusion_at_pos[2, i - 1, N - 1] * p_now[
            i - 1, N - 1]) / (2.0 * dx)
                - (diffusion_at_pos[3, i, 0] * p_now[i, 0] - diffusion_at_pos[3, i, N - 2] * p_now[i, N - 2]) / (
                            2.0 * dx)
        )

        # for points with well defined neighbours
        for j in range(1, N - 1):
            # first component
            flux_array[0, i, j] = (
                    (drift_at_pos[0, i, j] * p_now[i, j])
                    - (diffusion_at_pos[0, i + 1, j] * p_now[i + 1, j] - diffusion_at_pos[0, i - 1, j] * p_now[
                i - 1, j]) / (2.0 * dx)
                    - (diffusion_at_pos[1, i, j + 1] * p_now[i, j + 1] - diffusion_at_pos[1, i, j - 1] * p_now[
                i, j - 1]) / (2.0 * dx)
            )
            # second component
            flux_array[1, i, j] = (
                    (drift_at_pos[1, i, j] * p_now[i, j])
                    - (diffusion_at_pos[2, i + 1, j] * p_now[i + 1, j] - diffusion_at_pos[2, i - 1, j] * p_now[
                i - 1, j]) / (2.0 * dx)
                    - (diffusion_at_pos[3, i, j + 1] * p_now[i, j + 1] - diffusion_at_pos[3, i, j - 1] * p_now[
                i, j - 1]) / (2.0 * dx)
            )


def flux_power_efficiency(target_dir):  # processing of raw data
    phase_shift = 0.0
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # if abs(psi_1) >= abs(psi_2):
            integrate_flux_X = empty(min_array.size)
            integrate_flux_Y = empty(min_array.size)
            integrate_power_X = empty(min_array.size)
            integrate_power_Y = empty(min_array.size)
            efficiency_ratio = empty(min_array.size)

            for Ecouple in Ecouple_array:
                for ii, phase_shift in enumerate(phase_array):
                    if num_minima1 == 10.0:
                        input_file_name = (
                                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200116_bioparameters" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    else:
                        input_file_name = (
                                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190924_no_vary_n1_3" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")

                    output_file_name = (
                                target_dir + "200128_biologicalnumbers/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")

                    print(
                        "Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " + f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")

                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                   phase_shift), usecols=(0, 3, 4, 5, 6, 7, 8))
                        N = int(sqrt(len(data_array)))
                        print(N)
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

                        flux_array = zeros((2, N, N))
                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array) / (dx * dx)

                        integrate_flux_X[ii] = (1. / (2 * pi)) * trapz(trapz(flux_array[0, ...], dx=dx, axis=1), dx=dx)
                        integrate_flux_Y[ii] = (1. / (2 * pi)) * trapz(trapz(flux_array[1, ...], dx=dx, axis=0), dx=dx)

                        integrate_power_X[ii] = integrate_flux_X[ii] * psi_1
                        integrate_power_Y[ii] = integrate_flux_Y[ii] * psi_2
                    except:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                     phase_shift))
                if (abs(psi_1) <= abs(psi_2)):
                    efficiency_ratio = -(integrate_power_X / integrate_power_Y)
                else:
                    efficiency_ratio = -(integrate_power_Y / integrate_power_X)

                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                          "w") as ofile:
                    for ii, phase_shift in enumerate(phase_array):
                        ofile.write(
                            f"{phase_shift:.15e}" + "\t"
                            + f"{integrate_flux_X[ii]:.15e}" + "\t"
                            + f"{integrate_flux_Y[ii]:.15e}" + "\t"
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{efficiency_ratio[ii]:.15e}" + "\n")
                    ofile.flush()


def flux_power_efficiency_extrapoints(target_dir):  # processing of raw data
    phase_shift = 0.0
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if abs(psi_1) > abs(psi_2):
                integrate_flux_X = empty(phase_array.size)
                integrate_flux_Y = empty(phase_array.size)
                integrate_power_X = empty(phase_array.size)
                integrate_power_Y = empty(phase_array.size)
                efficiency_ratio = empty(phase_array.size)

                for Ecouple in Ecouple_array:
                    for ii, phase_shift in enumerate(phase_array):
                        if phase_shift in phase_array_test:
                            input_file_name = (
                                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                        else:
                            input_file_name = (
                                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                        # input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190610_phaseoffset_extra" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")

                        output_file_name = (
                                    target_dir + "191217_morepoints/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{4}_Ecouple_{5}" + "_outfile.dat")

                        print(
                            "Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " + f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")

                        try:
                            data_array = loadtxt(
                                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                       phase_shift), usecols=(0, 3, 4, 5, 6, 7, 8))
                            N = int(sqrt(len(data_array)))
                            print(N)
                            prob_ss_array = data_array[:, 0].reshape((N, N))
                            drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                            diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

                            flux_array = zeros((2, N, N))
                            calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                            flux_array = asarray(flux_array) / (dx * dx)

                            integrate_flux_X[ii] = (1. / (2 * pi)) * trapz(trapz(flux_array[0, ...], dx=dx, axis=1),
                                                                           dx=dx)
                            integrate_flux_Y[ii] = (1. / (2 * pi)) * trapz(trapz(flux_array[1, ...], dx=dx, axis=0),
                                                                           dx=dx)

                            integrate_power_X[ii] = integrate_flux_X[ii] * psi_1
                            integrate_power_Y[ii] = integrate_flux_Y[ii] * psi_2
                        except:
                            print('Missing file')
                    if (abs(psi_1) <= abs(psi_2)):
                        efficiency_ratio = -(integrate_power_X / integrate_power_Y)
                    else:
                        efficiency_ratio = -(integrate_power_Y / integrate_power_X)

                    with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), "w") as ofile:
                        for ii, phase_shift in enumerate(phase_array):
                            ofile.write(
                                f"{phase_shift:.15e}" + "\t"
                                + f"{integrate_flux_X[ii]:.15e}" + "\t"
                                + f"{integrate_flux_Y[ii]:.15e}" + "\t"
                                + f"{integrate_power_X[ii]:.15e}" + "\t"
                                + f"{integrate_power_Y[ii]:.15e}" + "\t"
                                + f"{efficiency_ratio[ii]:.15e}" + "\n")
                        ofile.flush()


def plot_power_phi_grid(
        target_dir):  # grid of plots of the power as a function of the phase offset, different plots are for different forces
    output_file_name = (target_dir + "power_Y_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='all')

    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "190624_Twopiweep_complete_set/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 3, 4))
                    # print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:, 0]
                    power_x_array = data_array[:, 1]
                    power_y_array = data_array[:, 2]

                    axarr[i, j].plot(phase_array, power_y_array, color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')
                    # plt.legend(Ecouple_array, title="Ecouple")
    f.text(0.5, 0.04, '$\phi$', ha='center')
    f.text(0.04, 0.5, 'Output power', va='center', rotation='vertical')
    plt.xticks(ticklst, ticklabels)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_power_phi_single(target_dir):  # plot of the power as a function of the phase offset

    output_file_name = (target_dir + "Power_ATP_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # Power subsystem 2
            plt.figure()
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.axhline(0, color='black', linewidth=1)

            # zero-barrier theory lines
            input_file_name = (
                    target_dir + "190624_Twopisweep_complete_set/processed_data/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            # Ecouple_array3 = array(data_array[:, 0])
            flux_x_array = array(data_array[:, 1])
            flux_y_array = array(data_array[:, 2])
            # power_x = flux_x_array * psi_1
            power_y = -flux_y_array * psi_2
            ax.axhline(power_y[17], color='C0', linewidth=2, label=None)

            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 3, 4))
                    phase_array = data_array[:, 0]
                    power_x_array = data_array[:, 1]
                    power_y_array = -data_array[:, 2]

                    ax.plot(phase_array, power_y_array, linestyle='-', marker='o', label=f'${Ecouple}$', markersize=8,
                            linewidth=2, color=colorlst[ii])
                except OSError:
                    print('Missing file')
                    # Infinite coupling result
            input_file_name = (target_dir + "190530_Twopisweep/master_output_dir/processed_data/"
                               + "Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
            data_array = loadtxt(input_file_name, usecols=(0, 1))
            phase_array = data_array[:, 0]
            flux_array = -psi_2 * data_array[:, 1]

            ax.plot(phase_array[:61], flux_array[:61], '-', label=f'$\infty$', linewidth=2, color='C6')
            ax.tick_params(axis='both', labelsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.yaxis.offsetText.set_fontsize(14)

            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(handles[::-1], labels[::-1], title=r'$\beta E_{\rm couple}$', fontsize=14, loc=[0.8, 0.1],
                            frameon=False)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)
            # plt.xlabel(r'$\phi\ (\rm rad)$', fontsize=20)
            f.text(0.55, 0.02, r'$\phi\ (\rm rev)$', fontsize=20, ha='center')
            plt.ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', fontsize=20)
            plt.xticks(ticklst, label_lst)
            plt.yticks([-0.0004, -0.0002, 0, 0.0002, 0.0004])
            # plt.ylim(-0.00035, 0.0005)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            f.tight_layout()
            f.subplots_adjust(bottom=0.12)
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            # plt.close()


def plot_power_efficiency_phi_single(target_dir):  # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "power_efficiency_phi_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 6))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # flux plot
            axarr[0].axhline(0, color='black', linewidth=1)  # line at zero

            # zero-barrier theory lines
            input_file_name = (
                        target_dir + "190624_Twopisweep_complete_set/processed_data/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array3 = array(data_array[:, 0])
            flux_x_array = array(data_array[:, 1])
            flux_y_array = array(data_array[:, 2])
            power_x = flux_x_array * psi_1
            power_y = -flux_y_array * psi_2
            axarr[0].axhline(power_y[17], color='C0', linewidth=2, label='$0$')

            # General data
            E0 = 2.0
            E1 = 2.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 1, 2))
                    phase_array = array(data_array[:, 0])
                    flux_x_array = array(data_array[:, 1])
                    flux_y_array = array(data_array[:, 2])
                except OSError:
                    print('Missing file flux')
            power_x = flux_x_array * psi_1
            power_y = -flux_y_array * psi_2
            # axarr[0].plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
            axarr[0].plot(phase_array, power_y, 'o', color='C1', label='$2$', markersize=8)

            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # axarr[0].set_yticks(ylabels_flux)
            axarr[0].yaxis.offsetText.set_fontsize(14)
            axarr[0].tick_params(axis='both', labelsize=14)
            axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', fontsize=20)
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            axarr[0].set_ylim((0, None))
            #
            # leg = axarr[0].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
            # leg_title = leg.get_title()
            # leg_title.set_fontsize(14)

            #########################################################    
            # efficiency plot
            axarr[1].axhline(0, color='black', linewidth=1)
            axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')
            axarr[1].set_aspect(0.5)

            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "191217_morepoints/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(5))
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
            axarr[1].set_yticks(ylabels_eff)
            axarr[1].set_xticks(ticklst)
            axarr[1].set_xticklabels(label_lst)
            # axarr[1].set_xlabel(r'$\phi$', fontsize=20)

            leg = axarr[1].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)

            f.text(0.55, 0.07, r'$\phi\ (\rm rev)$', fontsize=20, ha='center')
            f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
            f.text(0.03, 0.37, r'$\mathbf{b)}$', fontsize=20)
            f.tight_layout()
            # f.subplots_adjust(bottom=0.1)
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            # f.close()


def plot_efficiency_phi_single(target_dir):  # plot of the efficiency as a function of phase offset
    output_file_name = (target_dir + "Efficiency_phi_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax = plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "190530_Twopisweep/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0, 5))
                    phase_array = data_array[:, 0]
                    eff_array = data_array[:, 1]

                    plt.plot(phase_array, eff_array, 'o', color=plt.cm.cool(colorlist[ii]), label=f'{Ecouple}')
                except OSError:
                    print('Missing file')
            # infinite coupling limit
            input_file_name = (
                        target_dir + "190530_Twopisweep/master_output_dir/processed_data/" + "Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
            data_array = loadtxt(input_file_name, usecols=(0, 1))
            phase_array = data_array[:, 0]
            eff_array = -psi_2 / psi_1 * ones(len(phase_array))

            ax.plot(phase_array, eff_array, '-', color=plt.cm.cool(colorlist[3]), label=f'$\infty$')

            plt.legend(title="$E_{couple}$", loc='upper left')
            plt.xlabel('$\phi$')
            plt.ylabel('$\eta$')
            plt.xticks(ticklst, ticklabels)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_efficiency_Ecouple_single(target_dir):  # plot of the efficiency as a function of the coupling strength
    output_file_name = (target_dir + "efficiency_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax = plt.subplot(111)
            ax.axhline(0, color='black', linewidth=1)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))

            # Zero barrier result
            input_file_name = (
                        "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
            try:
                print(input_file_name.format(psi_1, psi_2))
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))

                Ecouple_array2 = array(data_array[1:, 0])
                Ecouple_array2 = append(Ecouple_array2, 128.0)  # add point to get a longer curve
                flux_x_array = array(
                    data_array[1:, 1])  # 1: is to skip the point at zero, which is problematic on a log scale
                flux_y_array = array(data_array[1:, 2])
                flux_x_array = append(flux_x_array, flux_x_array[-1])  # copy last point to add one
                flux_y_array = append(flux_y_array, flux_y_array[-1])

                if abs(psi_1) > abs(psi_2):
                    ax.plot(Ecouple_array2, -psi_2 * flux_y_array / (psi_1 * flux_x_array), '-',
                            color=plt.cm.cool(0.99), linewidth=1.0)
                elif abs(psi_2) > abs(psi_1):
                    ax.plot(Ecouple_array2, -psi_1 * flux_x_array / (psi_2 * flux_y_array), '-',
                            color=plt.cm.cool(0.99), linewidth=1.0)
            except:
                print('Missing data')

            # General results
            eff_array = []
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (target_dir + "200116_biologicalnumbers/processed_data/flux_power_efficiency_" +
                                   "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(5))
                    eff_array = append(eff_array, data_array[0])
                except OSError:
                    print('Missing file')
            plt.plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(0))
            # plt.legend(title="$E_{couple}$", loc='upper left') 
            plt.xlabel('$E_{couple}$')
            plt.ylabel('$\eta$')
            plt.xscale('log')
            plt.ylim((None, 1))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # plt.xticks(ticklst, ticklabels)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_efficiency_Ecouple_grid(target_dir):  # grid of plots of the efficiency as a function of the coupling strength
    output_file_name = (target_dir + "efficiency_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='all')

    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            axarr[i, j].axhline(0, color='black', linewidth=1)  # plot line at zero
            if abs(psi_1) > abs(psi_2):
                axarr[i, j].axhline(-psi_2 / psi_1, linestyle='--', color='grey', linewidth=1)  # plot line at zero
            elif abs(psi_2) > abs(psi_1):
                axarr[i, j].axhline(-psi_1 / psi_2, linestyle='--', color='grey', linewidth=1)  # plot line at zero

            for ii, Ecouple in enumerate(Ecouple_array):
                ##plot zero barrier theory
                input_file_name = (
                            "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
                try:
                    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                    Ecouple_array2 = array(data_array[1:, 0])
                    Ecouple_array2 = append(Ecouple_array2, 128.0)  # add point to get a longer curve
                    flux_x_array = array(data_array[1:, 1])
                    flux_y_array = array(data_array[1:, 2])
                    flux_x_array = append(flux_x_array, flux_x_array[-1])
                    flux_y_array = append(flux_y_array, flux_y_array[-1])
                    if abs(psi_1) > abs(psi_2):
                        axarr[i, j].plot(Ecouple_array2, -psi_2 * flux_y_array / (psi_1 * flux_x_array), '-',
                                         color=plt.cm.cool(0.99), linewidth=1.0)
                    elif abs(psi_2) > abs(psi_1):
                        axarr[i, j].plot(Ecouple_array2, -psi_1 * flux_x_array / (psi_2 * flux_y_array), '-',
                                         color=plt.cm.cool(0.99), linewidth=1.0)
                except:
                    print('Missing data')

            ##plot simulations zero barrier
            E0 = 0.0
            E1 = 0.0
            input_file_name = (
                        "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            eff_array = []
            for ii, Ecouple in enumerate(Ecouple_array):
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0, 5))
                    print('Ecouple=%f' % Ecouple)
                    phase = data_array[0]
                    eff = data_array[1]
                    if i == j:
                        eff = -1
                    eff_array.append(eff)
                except OSError:
                    print('Missing file')

            ##plot simulations 
            E0 = 2.0
            E1 = 2.0
            input_file_name = (
                        "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/190624_Twopisweep_complete_set/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            eff_array1 = []
            for ii, Ecouple in enumerate(Ecouple_array):
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0, 5))
                    print('Ecouple=%f' % Ecouple)
                    phase = data_array[0, 0]
                    eff = data_array[0, 1]
                    if i == j:
                        eff = -1
                    eff_array1.append(eff)
                except OSError:
                    print('Missing file')
            axarr[i, j].plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(0.99), markersize=4)
            axarr[i, j].plot(Ecouple_array, eff_array1, 'o', color=plt.cm.cool(0), markersize=4)
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
    # plt.legend(Ecouple_array, title="Ecouple")
    plt.ylim(-0.33, 1.0)
    plt.xscale('log')
    f.text(0.5, 0.12, '$E_{couple}$', ha='center')
    f.text(0.11, 0.52, '$\eta$', va='center', rotation='vertical')
    f.text(0.07, 0.74, '$\mu_{H^+}=1.0$', ha='center')
    f.text(0.06, 0.51, '2.0', ha='center')
    f.text(0.06, 0.29, '4.0', ha='center')
    f.text(0.3, 0.87, '$\mu_{ATP}=-1.0$', ha='center')
    f.text(0.53, 0.87, '$-2.0$', ha='center')
    f.text(0.77, 0.87, '$-4.0$', ha='center')
    f.tight_layout(pad=6.0, w_pad=1.0, h_pad=1.0)
    # plt.xticks(ticklst, ticklabels)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
    plt.close()


def plot_flux_phi_grid(target_dir):  # grid of plots of the flux as a function of the phase offset
    output_file_name = (target_dir + "flux_y_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='all')
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 1, 2))
                    # print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:, 0]
                    flux_x_array = data_array[:, 1]
                    flux_y_array = data_array[:, 2]

                    axarr[i, j].plot(phase_array, flux_y_array, color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')
                    # plt.legend(Ecouple_array, title="Ecouple")
    f.text(0.5, 0.04, '$\phi$', ha='center')
    f.text(0.04, 0.5, '$J_y$', va='center', rotation='vertical')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(ticklst, ticklabels)
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_power_Ecouple_grid(target_dir):  # grid of plots of the flux as a function of the phase offset
    output_file_name = (target_dir + "power_ATP_Ecouple_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='row', figsize=(8, 6))
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            # for i, psi_2 in enumerate(psi2_array):
            psi_2 = -psi_1 / ratio
            print(psi_1, psi_2)

            # line at highest Ecouple power
            input_file_name = (
                        target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, 128.0),
                                     usecols=(3, 4))
                if len(data_array) > 2:
                    power_x = array(data_array[0, 0])
                    power_y = -array(data_array[0, 1])
                else:
                    power_x = array(data_array[0])
                    power_y = -array(data_array[1])
                axarr[i, j].axhline(power_y, color='grey', linestyle=':', linewidth=1)
            except OSError:
                print('Missing file flux')

            # line at zero power
            axarr[i, j].axhline(0, color='black', linewidth=1)

            # zero-barrier result
            input_file_name = (
                        target_dir + "191217_morepoints/processed_data/" + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_x_array = array(data_array[:, 1])
            flux_y_array = array(data_array[:, 2])
            power_x = flux_x_array * psi_1
            power_y = -flux_y_array * psi_2

            axarr[i, j].plot(Ecouple_array2, power_y, '-', color='C0', linewidth=3)

            # E0=E1=2 barrier data
            power_x_array = []
            power_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (
                            target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(3, 4))

                    if len(data_array) > 2:
                        power_x = array(data_array[0, 0])
                        power_y = -array(data_array[0, 1])
                    else:
                        power_x = array(data_array[0])
                        power_y = -array(data_array[1])
                    power_x_array = append(power_x_array, power_x)
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file flux')
            axarr[i, j].plot(Ecouple_array_tot, power_y_array, '.', color='C1', markersize=14)

            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].spines['bottom'].set_visible(False)
            axarr[i, j].tick_params(axis='both', labelsize=14)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((1.6, 150))
            elif j == 1:
                axarr[i, j].set_xlim((2.3, 150))
            else:
                axarr[i, j].set_xlim((3.4, 150))

            if i == 0:
                axarr[i, j].set_ylim((-0.00004, 0.00009))
                axarr[i, j].set_yticks([-0.00004, 0, 0.00004, 0.00008])
                axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                # axarr[i, j].set_yticklabels([r'$-5.$', r'$0.$', r'$5.$'])
            elif i == 1:
                axarr[i, j].set_ylim((-0.00008, 0.00035))
                axarr[i, j].set_yticks([0, 0.00015, 0.0003])
                # axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                axarr[i, j].set_yticklabels([r'$0$', r'$15$', r'$30$'])
            else:
                axarr[i, j].set_ylim((-0.00045, 0.0014))
                axarr[i, j].set_yticks([0, 0.0005, 0.001])
                # axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                axarr[i, j].set_yticklabels([r'$0$', r'$50$', r'$100$'])

            if j == 0 and i > 0:
                axarr[i, j].yaxis.offsetText.set_fontsize(0)
            else:
                axarr[i, j].yaxis.offsetText.set_fontsize(14)

            if j == psi1_array.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi_ratio[::-1][i], labelpad=16, rotation=270, fontsize=14)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.0f$' % psi1_array[::-1][j], fontsize=14)

    f.tight_layout()
    f.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=20)
    f.text(0.01, 0.5, r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', va='center', rotation='vertical',
           fontsize=20)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=20)
    # f.text(0.5, 0.95, r'$2 \pi \beta \mu_{\rm H^+}\ (\rm rev^{-1})$', ha='center', fontsize=20)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (\rm k_{\rm B} T / rad)$', va='center', rotation=270, fontsize=20)

    f.text(0.12, 0.88, r'$\mathbf{a)}$', ha='center', fontsize=14)
    f.text(0.4, 0.88, r'$\mathbf{b)}$', ha='center', fontsize=14)
    f.text(0.67, 0.88, r'$\mathbf{c)}$', ha='center', fontsize=14)
    f.text(0.12, 0.6, r'$\mathbf{d)}$', ha='center', fontsize=14)
    f.text(0.4, 0.6, r'$\mathbf{e)}$', ha='center', fontsize=14)
    f.text(0.67, 0.6, r'$\mathbf{f)}$', ha='center', fontsize=14)
    f.text(0.12, 0.32, r'$\mathbf{g)}$', ha='center', fontsize=14)
    f.text(0.4, 0.32, r'$\mathbf{h)}$', ha='center', fontsize=14)
    f.text(0.67, 0.32, r'$\mathbf{i)}$', ha='center', fontsize=14)

    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_flux_phi_single(target_dir):  # plot of the flux as a function of the phase offset
    output_file_name = (target_dir + "flux_phi_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax = plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 1, 2))
                    phase_array = data_array[:, 0]
                    flux_x_array = data_array[:, 1]
                    flux_y_array = data_array[:, 2]
                    ax.plot(phase_array, flux_x_array, 'o', color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array, flux_y_array, 'v', color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')

                    ##infinite coupling data
            input_file_name = (target_dir + "Twopisweep/processed_data/Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
            data_array = loadtxt(input_file_name, usecols=(0, 1))
            phase_array = data_array[:, 0]
            flux_array = data_array[:, 1]

            ax.plot(phase_array, flux_array, '-', color=plt.cm.cool(colorlist[3]), label=f'$\infty$')

            plt.legend(title="$E_{couple}$", loc='upper left')
            plt.xlabel('$\phi$')
            plt.ylabel('Flux')
            plt.xticks(ticklst, ticklabels)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_flux_Ecouple_single(target_dir):  # plot of the flux as a function of the coupling strength
    output_file_name = (target_dir + "flux_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if abs(psi_1) >= abs(psi_2):
                plt.figure()
                ax = plt.subplot(111)
                ax.axhline(0, color='black', linewidth=1)  # line at zero

                # zero-barrier theory lines
                # input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
                # data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                # Ecouple_array2 = array(data_array[:,0])
                # Ecouple_array2 = append(Ecouple_array2, 128.0)
                # flux_x_array = array(data_array[:,1])
                # flux_y_array = array(data_array[:,2])
                # flux_x_array = append(flux_x_array, flux_x_array[-1])
                # flux_y_array = append(flux_y_array, flux_y_array[-1])
                # plt.plot(Ecouple_array2, flux_x_array, '--', color=plt.cm.cool(.99))
                # plt.plot(Ecouple_array2, flux_y_array, '-', color=plt.cm.cool(.99))

                # #FP zero-barrier data points
                # flux_x_array=[]
                # flux_y_array=[]
                # E0=0.0
                # E1=0.0
                # for ii, Ecouple in enumerate(Ecouple_array):
                #     input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                #     try:
                #         data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                #         flux_x = data_array[1]
                #         flux_y = data_array[2]
                #         flux_x_array.append(flux_x)
                #         flux_y_array.append(flux_y)
                #     except OSError:
                #         print('Missing file')
                # plt.plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(.99))#, label=label_lst[i]
                # plt.plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(.99))

                # General data
                for j, num in enumerate(min_array):
                    num_minima1 = num
                    num_minima2 = 3.0
                    flux_x_array = []
                    flux_y_array = []
                    for ii, Ecouple in enumerate(Ecouple_array):
                        input_file_name = (targt_dir + "200116_biologicalnumbers/processed_data/" +
                                           "flux_power_efficiency_" +
                                           "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple),
                                                 usecols=(0, 1, 2))
                            flux_x = data_array[j, 1]
                            flux_y = data_array[j, 2]
                            flux_x_array.append(flux_x)
                            flux_y_array.append(flux_y)
                        except OSError:
                            print('Missing file')
                    if num == 3.0:
                        plt.axhline(y=flux_y_array[-1], ls='--', color='grey',
                                    linewidth=1)  # dashed line to emphasize peak in flux
                    plt.plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(colorlist[j]), markersize=size_lst[j],
                             label=num)
                    plt.plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(colorlist[j]), markersize=size_lst[j])

                # add in a extra data points
                # flux_x_array=[]
                # flux_y_array=[]
                # E0=2.0
                # E1=2.0
                # for ii, Ecouple in enumerate(Ecouple_array_extra):
                #     input_file_name = (target_dir + "190610_Extra_measurements/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                #     try:
                #         data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                #         flux_x = data_array[i,1]
                #         flux_y = data_array[i,2]
                #         flux_x_array.append(flux_x)
                #         flux_y_array.append(flux_y)
                #     except OSError:
                #         print('Missing file')
                # plt.plot(Ecouple_array_extra, flux_x_array, 'o', color=plt.cm.cool(colorlist[0]))
                # plt.plot(Ecouple_array_extra, flux_y_array, 'v', color=plt.cm.cool(colorlist[0]))

                plt.legend(title="$n_1$")
                plt.xlabel('$E_{couple}$')
                plt.ylabel('Flux')
                plt.xscale('log')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phase_array[i]))
                plt.close()


def plot_flux_Ecouple_grid(target_dir):  # grid of plots of the flux as a function of the coupling strength
    output_file_name = (target_dir + "flux_Ecouple_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='all')
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            axarr[i, j].axhline(0, color='black', linewidth=1)  # plot line at zero

            ##plot zero barrier theory
            try:
                input_file_name = (
                            "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                Ecouple_array2 = array(data_array[1:, 0])
                Ecouple_array2 = append(Ecouple_array2, 128.0)  # add point to get a longer curve
                flux_x_array = array(data_array[1:, 1])
                flux_y_array = array(data_array[1:, 2])
                flux_x_array = append(flux_x_array, flux_x_array[-1])
                flux_y_array = append(flux_y_array, flux_y_array[-1])
                axarr[i, j].plot(Ecouple_array2, flux_x_array, '--', color=plt.cm.cool(0.99), linewidth=1.0)
                axarr[i, j].plot(Ecouple_array2, flux_y_array, '-', color=plt.cm.cool(0.99), linewidth=1.0)
            except:
                print('Missing data')

            ##plot zero barrier data
            flux_x_array = []
            flux_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/processed_data/" + "flux_power_efficiency_" + "E0_0.0_E1_0.0_psi1_{0}_psi2_{1}_n1_{2}_n2_{3}_Ecouple_{4}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                                         usecols=(0, 1, 2))
                    # phase = data_array[0]
                    flux_x = data_array[1]
                    flux_x_array.append(flux_x)
                    flux_y = data_array[2]
                    flux_y_array.append(flux_y)
                except OSError:
                    print('Missing file')
            axarr[i, j].plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(0.99), markersize=3)
            axarr[i, j].plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(0.99), markersize=3)
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)

            ##General data
            flux_x_array = []
            flux_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 1, 2))
                    flux_x = data_array[0, 1]
                    flux_x_array.append(flux_x)
                    flux_y = data_array[0, 2]
                    flux_y_array.append(flux_y)
                except OSError:
                    print('Missing file')

            axarr[i, j].plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(0), markersize=4)
            axarr[i, j].plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(0), markersize=4)
            plt.xscale('log')
            axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)

    f.text(0.52, 0.09, '$E_{couple}$', ha='center')
    f.text(0.09, 0.51, 'J', va='center', rotation='vertical')
    f.text(0.07, 0.75, '$\mu_{H^+}=1.0$', ha='center')
    f.text(0.06, 0.5, '2.0', ha='center')
    f.text(0.06, 0.27, '4.0', ha='center')
    f.text(0.3, 0.9, '$\mu_{ATP}=-1.0$', ha='center')
    f.text(0.53, 0.9, '$-2.0$', ha='center')
    f.text(0.77, 0.9, '$-4.0$', ha='center')
    plt.xscale('log')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.xticks(ticklst, ticklabels)
    f.tight_layout(pad=5.0, w_pad=1.0, h_pad=1.0)
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
    plt.close()


def plot_flux_contour(target_dir):  # contourplot of the flux as a function as the states of the two subsystems
    psi_1 = 0.0
    psi_2 = 0.0
    Ecouple = 0.0
    phase = 0.0
    flux_array = empty((2, N, N))
    input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset/" + "reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    output_file_name = (
                target_dir + "flux_theta0_theta1_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}_phase_{7}" + "_.pdf")

    try:
        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase),
                             usecols=(0, 3, 4, 5, 6, 7, 8))
        prob_ss_array = data_array[:, 0].reshape((N, N))
        drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
        diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
        flux_array = asarray(flux_array) / (dx * dx)
        flux_x_array = flux_array[0]
        flux_y_array = flux_array[1]

        plt.contourf(positions, positions, flux_x_array)
        plt.colorbar()
    except OSError:
        print('Missing file')
    plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple, phase))
    plt.close()


def plot_flux_space(target_dir):  # plot of the integrated flux as a function of the position
    psi_1 = 0.0
    psi_2 = 0.0
    phase_shift = 0.0
    Ecouple = 0.0
    input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset/" + "reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    output_file_name = (
                target_dir + "flux_theta_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}_phase_{7}" + "_outfile.dat")
    integrate_flux_X = zeros(N)
    integrate_flux_Y = zeros(N)

    print(
        "Calculating flux for " + f"psi_2 = {psi_2}, psi_1 = {psi_1}, " + f"Ecouple = {Ecouple}, phase = {phase_shift}")
    flux_array = empty((2, N, N))
    data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift),
                         usecols=(0, 3, 4, 5, 6, 7, 8))
    prob_ss_array = data_array[:, 0].reshape((N, N))
    drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
    diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

    calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)

    flux_array = asarray(flux_array) / (dx * dx)

    integrate_flux_X = trapz(flux_array[0, ...], dx=dx, axis=1)
    integrate_flux_Y = trapz(flux_array[1, ...], dx=dx, axis=0)

    plt.plot(positions, integrate_flux_X)
    plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple, phase))
    plt.close()


def plot_power_efficiency_Ecouple_single(
        target_dir):  # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "power_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 8))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # flux plot
            axarr[0].axhline(0, color='black', linewidth=0.5)  # line at zero
            maxpower = 0.000085247
            # axarr[0].axhline(maxpower, color='grey', linestyle=':', linewidth=1)#line at infinite power coupling (calculated in Mathematica)
            # axarr[0].axhline(1, color='grey', linestyle=':', linewidth=1)#line at infinite power coupling
            # axarr[0].axvline(12, color='grey', linestyle=':', linewidth=1)#lining up features in the two plots

            # zero-barrier theory lines
            input_file_name = (target_dir + "190624_Twopisweep_complete_set/processed_data/"
                               + "Flux_zerobarrier_evenlyspaced_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_x_array = array(data_array[:, 1])
            flux_y_array = array(data_array[:, 2])
            power_x = flux_x_array * psi_1
            power_y = -flux_y_array * psi_2
            # axarr[0].plot(Ecouple_array2, power_x, '--', color=plt.cm.cool(.99))
            axarr[0].plot(Ecouple_array2, power_y, '-', color='C0', label='$0$', linewidth=2)

            # General data
            i = 0  # only use phase=0 data
            power_x_array = []
            power_y_array = []
            E0 = 2.0
            E1 = 2.0
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 3, 4))
                    if Ecouple in Ecouple_array:
                        power_x = array(data_array[i, 1])
                        power_y = array(data_array[i, 2])
                    else:
                        power_x = array(data_array[1])
                        power_y = array(data_array[2])
                    power_x_array = append(power_x_array, power_x)
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file flux')
            # axarr[0].plot(Ecouple_array_tot, -power_y_array, 'o', color='C1', label='$2$', markersize=8)

            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0].yaxis.offsetText.set_fontsize(14)
            # axarr[0].set_yticks(ylabels_flux)
            # axarr[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axarr[0].tick_params(axis='y', labelsize=14)
            axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (t_{\rm sim}^{-1}) $', fontsize=20)
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            axarr[0].spines['bottom'].set_visible(False)
            axarr[0].set_xlim((1.7, 135))

            leg = axarr[0].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)

            #########################################################    
            # efficiency plot
            axarr[1].axhline(0, color='black', linewidth=0.5)
            axarr[1].axvline(12, color='grey', linestyle=':', linewidth=1)
            # ax.axhline(0.5, color='grey', linestyle='--', linewidth=1) 
            input_file_name = (
                        target_dir + "190624_Twopisweep_complete_set/processed_data/" + "Flux_zerobarrier_evenlyspaced_psi1_{0}_psi2_{1}_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                Ecouple_array2 = array(data_array[1:, 0])
                Ecouple_array2 = append(Ecouple_array2, 128.0)  # add point to get a longer curve
                flux_x_array = array(
                    data_array[1:, 1])  # 1: is to skip the point at zero, which is problematic on a log scale
                flux_y_array = array(data_array[1:, 2])
                flux_x_array = append(flux_x_array, flux_x_array[-1])  # copy last point to add one
                flux_y_array = append(flux_y_array, flux_y_array[-1])

                if abs(psi_1) > abs(psi_2):
                    axarr[1].plot(Ecouple_array2, flux_y_array / (flux_x_array), '-', color='C0', linewidth=2)
                elif abs(psi_2) > abs(psi_1):
                    axarr[1].plot(Ecouple_array2, flux_x_array / (flux_y_array), '-', color='C0', linewidth=2)
            except:
                print('Missing data efficiency')

            eff_array = []
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (
                            target_dir + "191217_morepoints/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(5))
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
            # axarr[1].set_ylim((None,))
            axarr[1].set_xlim((1.7, 135))
            axarr[1].spines['right'].set_visible(False)
            axarr[1].spines['top'].set_visible(False)
            # axarr[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            axarr[1].spines['bottom'].set_visible(False)
            axarr[1].set_yticks(ylabels_eff)
            axarr[1].tick_params(axis='both', labelsize=14)

            f.text(0.05, 0.95, r'$\mathbf{a)}$', ha='center', fontsize=20)
            f.text(0.05, 0.48, r'$\mathbf{b)}$', ha='center', fontsize=20)
            f.subplots_adjust(hspace=0.01)
            f.tight_layout()
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_power_Ecouple_single(target_dir):  # plot of power as a function of coupling strength
    output_file_name = (target_dir + "power_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(6, 4))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            axarr.axhline(0, color='black', linewidth=1)  # line at zero
            # axarr.axhline(1.029 * 10 ** (-8), color='grey', linestyle='--', linewidth=1)  # line at infinite coupling

            # zero-barrier theory lines
            # input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/" +
            #                    "fokker_planck/working_directory_cython/200128_biologicalnumbers/processed_data/" +
            #                    "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            # data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            # Ecouple_array2 = array(data_array[:, 0])
            # flux_y_array = array(data_array[:, 2])
            # power_y = -flux_y_array*psi_2
            # axarr.plot(Ecouple_array2, power_y, '-')

            # power_x_array=[]
            power_y_array = []
            i = 0  # only use phase=0 data
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (target_dir + "200128_biologicalnumbers/processed_data/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple)
                                         , usecols=(3, 4))
                    # power_x = array(data_array[0])
                    power_y = array(data_array[1])
                    # power_x_array = append(power_x_array, power_x)
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file power')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
            # axarr[0].plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
            axarr.plot(Ecouple_array, -power_y_array, 'o')

            # add in a extra data points
            # flux_x_array=[]
            # flux_y_array=[]
            # for ii, Ecouple in enumerate(Ecouple_array_extra):
            #     input_file_name = (target_dir + "190610_Extra_measurements_Ecouple/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            #     try:
            #         data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
            #         flux_x = data_array[i,1]
            #         flux_y = data_array[i,2]
            #         flux_x_array = append(flux_x_array, flux_x)
            #         flux_y_array = append(flux_y_array, flux_y)
            #     except OSError:
            #         print('Missing file flux extra points')
            # power_x = flux_x_array*psi_1
            # power_y = flux_y_array*psi_2
            # # axarr.plot(Ecouple_array_extra, psi_1*flux_x_array, 'o', color=plt.cm.cool(colorlist[0]))
            # axarr.plot(Ecouple_array_extra, power_y, 'o', color=plt.cm.cool(colorlist[0]))

            axarr.set_xscale('log')
            axarr.set_xlabel('$\\beta E_{\\rm couple}$')
            axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # axarr.set_yticks(ylabels_flux)
            axarr.set_ylabel('$\\mathcal{P}_{\\rm ATP}$')
            axarr.spines['right'].set_visible(False)
            axarr.spines['top'].set_visible(False)
            # axarr.set_ylim((0, None))

            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_power_Ecouple_scaled(target_dir):  # plot of power as a function of coupling strength
    output_file_name = (
                target_dir + "power_scaled_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n_{4}" + "_log_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                f, axarr = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(6, 4))
                axarr.axhline(0, color='black', linewidth=1)  # line at zero
                axarr.axhline(1, color='grey', linestyle='--', linewidth=1)  # line to emphasize peak

                # zero-barrier theory lines
                input_file_name = (
                            "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
                try:
                    # print(input_file_name.format(psi_1, psi_2))
                    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                    Ecouple_array2 = array(data_array[:, 0])
                    Ecouple_array2 = append(Ecouple_array2, 128.0)
                    flux_x_array = array(data_array[:, 1])
                    flux_y_array = array(data_array[:, 2])
                    flux_x_array = append(flux_x_array, flux_x_array[-1])
                    flux_y_array = append(flux_y_array, flux_y_array[-1])
                    power_x = flux_x_array * psi_1
                    power_y = flux_y_array * psi_2
                    fluxinf = 0.5 * (psi_1 + psi_2) / (2 * pi * 1000)
                    power_inf_x = fluxinf * psi_1
                    power_inf_y = fluxinf * psi_2
                    # axarr.plot(Ecouple_array2, power_x, '--', color=plt.cm.cool(.99))
                    axarr.plot(Ecouple_array2, power_y / power_inf_y, '-', color=plt.cm.cool(.99))
                except:
                    print('Zero barrier data unavailable')

                # Infinite coupling
                input_file_name = (
                            target_dir + "190624_Twopisweep_complete_set/processed_data/" + "flux_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_inf" + "_outfile.dat")
                try:
                    # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                    phase = array(data_array[0, 0])
                    flux = array(data_array[0, 1])
                except:
                    print('Missing file infinite coupling')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                power_inf_x = flux * psi_1
                power_inf_y = flux * psi_2

                # General data
                flux_x_array = zeros(len(Ecouple_array))
                flux_y_array = zeros(len(Ecouple_array))
                i = 0  # only use phase=0 data
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "190624_Twopisweep_complete_set/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                            usecols=(0, 1, 2))
                        flux_x = data_array[i, 1]
                        flux_y = data_array[i, 2]
                        flux_x_array[ii] = flux_x
                        flux_y_array[ii] = flux_y
                    except OSError:
                        print('Missing file flux')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                power_x = flux_x_array * psi_1
                power_y = flux_y_array * psi_2
                # axarr.plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
                axarr.plot(Ecouple_array, power_y / power_inf_y, 'o', color=plt.cm.cool(0))

                # add in a extra data points
                flux_x_array = zeros(len(Ecouple_array_extra))
                flux_y_array = zeros(len(Ecouple_array_extra))
                c = 0
                for ii, Ecouple in enumerate(Ecouple_array_extra):
                    input_file_name = (
                                target_dir + "190610_Extra_measurements_Ecouple/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                            usecols=(0, 1, 2))
                        flux_x = data_array[i, 1]
                        flux_y = data_array[i, 2]
                        flux_x_array[ii] = flux_x
                        flux_y_array[ii] = flux_y
                        c = 0
                    except OSError:
                        print('Missing file flux extra points')
                        c = 1
                        # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                if c == 0:
                    power_x = flux_x_array * psi_1
                    power_y = flux_y_array * psi_2
                    # axarr.plot(Ecouple_array_extra, psi_1*flux_x_array, 'o', color=plt.cm.cool(colorlist[0]))
                    axarr.plot(Ecouple_array_extra, power_y / power_inf_y, 'o', color=plt.cm.cool(colorlist[0]))

                axarr.set_xscale('log')
                axarr.set_xlabel('$E_{couple}$')
                axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                # axarr.set_yticks(ylabels_flux)
                axarr.set_ylabel('$P_{ATP/ADP}/P_{ATP/ADP}^{\infty}$')
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                f.tight_layout()

                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1))
                plt.close()


def plot_energy_flux(target_dir):  # contourplot of the energy landscape with the 2D flux on top in the form of arrows
    psi_1 = 4.0
    psi_2 = -2.0
    Ecouple = 128.0
    phase = 0.0
    flux_array = empty((2, N, N))

    input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset/" + "reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    output_file_name = (
                target_dir + "Energy_flux_" + "Ecouple_{0}_E0_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_.pdf")

    fig, ax = plt.subplots()
    try:
        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase),
                             usecols=(0, 2, 3, 4, 5, 6, 7, 8))
        prob_ss_array = data_array[:, 0].reshape((N, N))
        pot_array = data_array[:, 1].reshape((N, N))
        drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
        diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))

        plt.contourf(positions, positions, pot_array)
        plt.colorbar()

        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)

        flux_array = asarray(flux_array) / (dx * dx)
        flux_x_array = flux_array[0]
        flux_y_array = flux_array[1]

        M = 36
        fluxX = empty((M, M))
        fluxY = empty((M, M))
        for i in range(M):
            fluxX[i] = flux_x_array[int(N / M) * i, ::int(N / M)]
            fluxY[i] = flux_y_array[int(N / M) * i, ::int(N / M)]

        plt.quiver(positions[::int(N / M)], positions[::int(N / M)], fluxX, fluxY,
                   units='xy')  # headlength=1, headwidth=1, headaxislength=1

    except OSError:
        print('Missing file')

    plt.xlabel('$\theta_o$')
    plt.ylabel('$\theta_1$')
    plt.xticks(ticklst, ticklabels)
    plt.yticks(ticklst, ticklabels)

    fig.savefig(output_file_name.format(Ecouple, E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
    plt.close()


def plot_energy_flux_grid(
        target_dir):  # grid of contourplots of the energy landscape with the 2D flux on top in the form of arrows

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # define arrays to add forces to energy landscapes
            Fx_array = empty((N, N))
            Fy_array = empty((N, N))
            Fx = psi_1 * positions
            Fy = psi_2 * positions
            for k in range(0, N):
                Fx_array[k] = Fx
                Fy_array[:, k] = Fy

            f, axarr = plt.subplots(3, 7, sharex='all', sharey='all', figsize=(12, 6))
            output_file_name = (
                        target_dir + "190530_Twopisweep/Energy_flux_grid_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = (
                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset/" + "reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            data_array = loadtxt(input_file_name.format(E0, 16.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                                 usecols=(0, 2, 3, 4, 5, 6, 7, 8))
            prob_ss_array = data_array[:, 0].reshape((N, N))
            pot_array = data_array[:, 1].reshape((N, N))
            minpot = amin(pot_array)
            maxpot = amax(pot_array)
            drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
            diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
            flux_array = empty((2, N, N))
            calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
            flux_array = asarray(flux_array) / (dx * dx)
            flux_length_array = empty((N, N))
            flux_x_array = flux_array[0]
            flux_y_array = flux_array[1]
            flux_length_array = flux_x_array * flux_x_array + flux_y_array * flux_y_array
            maxflux = amax(flux_length_array)
            print(maxpot, math.sqrt(maxflux))

            ##plotting
            for i, Ecouple in enumerate(Ecouple_array):
                for j, phase in enumerate(phase_array):
                    flux_array = empty((2, N, N))
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset/" + "reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")

                    try:
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase),
                            usecols=(0, 2, 3, 4, 5, 6, 7, 8))
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        pot_array = data_array[:, 1].reshape((N, N))
                        drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))

                        if i == 2 and j == 0:
                            im = axarr[i, j].contourf(positions, positions, (pot_array.T), vmin=minpot, vmax=maxpot,
                                                      cmap=plt.cm.cool)
                        else:
                            im2 = axarr[i, j].contourf(positions, positions, (pot_array.T), vmin=minpot, vmax=maxpot,
                                                       cmap=plt.cm.cool)

                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array) / (dx * dx)
                        flux_x_array = flux_array[0]
                        flux_y_array = flux_array[1]

                        # select fewer arrows to draw
                        M = 18  # number of arrows in a row/ column, preferably a number such that N/M is an integer.
                        fluxX = empty((M, M))
                        fluxY = empty((M, M))
                        for k in range(M):
                            fluxX[k] = flux_x_array[int(N / M) * k, ::int(N / M)]
                            fluxY[k] = flux_y_array[int(N / M) * k, ::int(N / M)]

                        axarr[i, j].quiver(positions[::int(N / M)], positions[::int(N / M)], fluxX.T, fluxY.T,
                                           units='xy', angles='xy', scale_units='xy', scale=math.sqrt(maxflux))
                        axarr[i, j].set_aspect(aspect=1, adjustable='box-forced')

                    except OSError:
                        print('Missing file')

            f.text(0.45, 0.04, '$X$', ha='center')
            f.text(0.05, 0.5, '$Y$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.xticks(ticklst, ticklabels)
            plt.yticks(ticklst, ticklabels)
            f.subplots_adjust(right=0.8)
            cbar_ax = f.add_axes([0.85, 0.25, 0.03, 0.5])
            cbar = f.colorbar(im, cax=cbar_ax)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_prob_flux(
        target_dir):  # contourplot of the 2D steady state probability with the flux on top in the form of arrows
    phase = 0.0
    Ecouple = 16.0
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            output_file_name = (target_dir + "Pss_" + "Ecouple_{0}_E0_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}"
                                + "_big_.pdf")

            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid,
            # so that we can scale the colors and arrows by it
            input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200116_bioparameters" + "/reference_"
                               + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                            phase), usecols=(0, 2, 3, 4, 5, 6, 7, 8))
                prob_ss_array = data_array[:, 0].reshape((N, N))
                maxprob = amax(prob_ss_array)
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
                flux_array = empty((2, N, N))
                calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                flux_array = asarray(flux_array) / (dx * dx)
                flux_length_array = empty((N, N))
                flux_x_array = flux_array[0]
                flux_y_array = flux_array[1]
                flux_length_array = flux_x_array * flux_x_array + flux_y_array * flux_y_array
                maxflux = amax(flux_length_array)
                print(maxprob, math.sqrt(maxflux))

                plt.contourf(positions, positions, prob_ss_array.T, vmin=0, vmax=maxprob, cmap=plt.cm.cool)

                # select fewer arrows to draw
                # M = 18 #number of arrows in a row/ column, preferably a number such that N/M is an integer.
                # fluxX = empty((M, M))
                # fluxY = empty((M, M))
                # for k in range(M):
                #     fluxX[k] = flux_x_array[int(N/M)*k, ::int(N/M)]
                #     fluxY[k] = flux_y_array[int(N/M)*k, ::int(N/M)]
                # fluxzeros = zeros((M, M))
                #
                # plt.quiver(positions[::int(N/M)], positions[::int(N/M)], fluxX.T, fluxY.T, units='xy', angles='xy', scale_units='xy', scale=math.sqrt(maxflux))

            except OSError:
                print('Missing file')
            plt.xlabel('$\\theta_{\\rm o}$')
            plt.ylabel('$\\theta_{1}$')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # plt.xticks(ticklst, ticklabels)
            # plt.yticks(ticklst, ticklabels)
            ax.set_aspect(aspect=1.0)
            plt.savefig(output_file_name.format(Ecouple, E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_prob_flux_grid(
        target_dir):  # grid of contourplots of the 2D equilibrium probability with the flux on top in the form of arrows

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            f, axarr = plt.subplots(1, 7, sharex='all', sharey='all', figsize=(12, 2))
            output_file_name = (target_dir + "Pss_y|x_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

            # determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" +
                               "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                               "_outfile.dat")
            print(input_file_name.format(E0, 64.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
            data_array = loadtxt(input_file_name.format(E0, 64.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
            prob_ss_array = data_array[:, 0].reshape((N, N))
            prob_ss_y = trapz(prob_ss_array, dx=1, axis=0)
            prob_ss_y = prob_ss_y.reshape((1, N))
            prob_ss_yx = prob_ss_array / prob_ss_y

            # prob_eq_array = data_array[:, 1].reshape((N,N))
            # pot_array = data_array[:, 2].reshape((N,N))
            maxprob = amax(prob_ss_yx)
            # maxpot=amax(pot_array)
            # drift_at_pos = data_array[:, 3:5].T.reshape((2,N,N))
            # diffusion_at_pos = data_array[:, 5:].T.reshape((4,N,N))
            # flux_array = empty((2,N,N))
            # calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
            # flux_array = asarray(flux_array)/(dx*dx)
            # flux_length_array = empty((N,N))
            # flux_x_array = flux_array[0]
            # flux_y_array = flux_array[1]
            # flux_length_array = flux_x_array*flux_x_array + flux_y_array*flux_y_array
            # maxflux=amax(flux_length_array)
            # print(maxpot, math.sqrt(maxflux))

            # plotting
            for i, Ecouple in enumerate(Ecouple_array):
                # for j, phase in enumerate(phase_array):
                phase = 0.0
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                flux_array = empty((2, N, N))
                try:
                    data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                                phase), usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
                    prob_ss_array = data_array[:, 0].reshape((N, N))
                    prob_eq_array = data_array[:, 1].reshape((N, N))
                    pot_array = data_array[:, 2].reshape((N, N))
                    drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
                    diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))
                    prob_ss_y = trapz(prob_ss_array, dx=1, axis=0)
                    prob_ss_y = prob_ss_y.reshape((1, N))
                    prob_ss_yx = prob_ss_array / prob_ss_y

                    im = axarr[i].contourf(positions, positions, prob_ss_yx, vmin=0, vmax=maxprob,
                                           cmap=plt.cm.cool)  # plot energy landscape
                    axarr[i].set_title('$%.1f$' % Ecouple)
                    axarr[i].set_xlim([0, 2 * pi])
                    axarr[i].set_ylim([0, 2 * pi])
                    # axarr[i,j].plot(positions, positions, color='grey', linewidth=1.0) #line on the diagonal

                    # calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                    # flux_array = asarray(flux_array)/(dx*dx)
                    # flux_x_array = (flux_array[0])
                    # flux_y_array = (flux_array[1])

                    # select fewer arrows to draw
                    # M = 18 #number of arrows in a row/ column, preferably a number such that N/M is an integer.
                    # fluxX = empty((M, M))
                    # fluxY = empty((M, M))
                    # for k in range(M):
                    #     fluxX[k] = flux_x_array[int(N/M)*k, ::int(N/M)]
                    #     fluxY[k] = flux_y_array[int(N/M)*k, ::int(N/M)]
                    # fluxzeros = zeros((M, M))
                    #
                    # axarr[i,j].quiver(positions[::int(N/M)], positions[::int(N/M)], fluxX.T, fluxY.T, units='xy', angles='xy', scale_units='xy', scale=math.sqrt(maxflux))
                    # axarr[i,j].set_aspect(aspect=1, adjustable='box-forced')

                except OSError:
                    print('Missing file')
            f.text(0.52, 0.94, '$E_{\\rm couple}$', ha='center')
            f.text(0.52, 0.01, '$\\theta_{\\rm o}$', ha='center')
            f.text(0.04, 0.5, '$\\theta_1$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.xticks(ticklst, ticklabels)
            plt.yticks(ticklst, ticklabels)

            f.tight_layout()
            top = 0.8
            bottom = 0.2
            right = 0.9
            left = 0.1
            f.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
            cbar_ax = f.add_axes([1.03 * right, bottom, 0.02, top - bottom])
            cbar = f.colorbar(im, cax=cbar_ax)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_condprob_grid(target_dir):  # grid of contourplots of the conditional, steady state probability

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            f, axarr = plt.subplots(3, 7, sharex='all', sharey='all', figsize=(12, 6))
            output_file_name = (
                        target_dir + "Condprob_10_flux_grid_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_big_low.pdf")

            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = (
                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190729_varying_n/" + "reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            data_array = loadtxt(input_file_name.format(E0, 16.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                                 usecols=(0))
            prob_ss_array = data_array.reshape((N, N))
            PXss = trapz(prob_ss_array, axis=1)  # axis=0 gives P(f1), change to axis=1 for P(fo)
            cond_prob_array = empty((N, N))
            for i in range(0, N):
                for j in range(0, N):
                    cond_prob_array[i, j] = prob_ss_array[i, j] / PXss[i]  # P(x2|x1)=P(x1,x2)/P(x1)
            maxprob = amax(cond_prob_array)
            print('maximum probability in grid:', maxprob)

            # actually making subplots
            for i, Ecouple in enumerate(Ecouple_array):
                for j, phase in enumerate(phase_array):
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase),
                            usecols=(0))
                        prob_ss_array = data_array.reshape((N, N))
                        PXss = trapz(prob_ss_array, axis=1)
                        cond_prob_array = empty((N, N))
                        for ii in range(0, N):
                            for jj in range(0, N):
                                cond_prob_array[ii, jj] = prob_ss_array[ii, jj] / PXss[ii]
                        if i == 2 and j == 0:
                            im = axarr[i, j].contourf(positions, positions, cond_prob_array.T, vmin=0, vmax=maxprob,
                                                      cmap=plt.cm.cool)
                        else:
                            im2 = axarr[i, j].contourf(positions, positions, cond_prob_array.T, vmin=0, vmax=maxprob,
                                                       cmap=plt.cm.cool)
                        axarr[i, j].set_aspect(aspect=1, adjustable='box-forced')
                    except OSError:
                        print('Missing file')
            f.text(0.5, 0.04, '$F_o$', ha='center')
            f.text(0.05, 0.5, '$F_1$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.xticks(ticklst, ticklabels)
            plt.yticks(ticklst, ticklabels)
            f.subplots_adjust(right=0.85)
            cbar_ax = f.add_axes([0.9, 0.25, 0.03, 0.5])
            cbar = f.colorbar(im, cax=cbar_ax)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            f.subplots_adjust(wspace=0.2, hspace=0.05)
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_rel_flux_Ecouple(
        target_dir):  # plot of the relative flux (flux F1 divided by flux Fo) as a function of the coupling strength
    output_file_name = (
                target_dir + "relflux_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax = plt.subplot(111)
            ax.axhline(0, color='black', linewidth=1)  # line at zero

            for i, phase in enumerate(phase_array):
                flux_x_array = []
                flux_y_array = []
                rel_flux = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        print(
                            "Plotting " + f"psi_2 = {psi_2}, psi_1 = {psi_1}, " + f"Ecouple = {Ecouple}, phase = {phase}")
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                            usecols=(0, 1, 2))
                        flux_x = data_array[i, 1]
                        flux_y = data_array[i, 2]
                        flux_x_array.append(flux_x)
                        flux_y_array.append(flux_y)
                    except OSError:
                        print('Missing file')
                try:
                    for k in range(0, len(Ecouple_array)):
                        rel_flux.append(flux_y_array[k] / flux_x_array[k])
                    plt.plot(Ecouple_array, rel_flux, 'o', color=plt.cm.cool(colorlist[i]), markersize=size_lst[i],
                             label=label_lst[i])
                except:
                    print('Missing data')

            plt.legend(title="$\phi$")
            plt.xlabel('$E_{couple}$')
            plt.ylabel('Relative flux')
            plt.xscale('log')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()


def plot_marg_prob_space(
        target_dir):  # plot of the relative flux (flux F1 divided by flux Fo) as a function of the coupling strength
    output_file_name1 = (
                target_dir + "margprob_x_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_.pdf")
    output_file_name2 = (
                target_dir + "margprob_y_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            for ii, Ecouple in enumerate(Ecouple_array):
                plt.figure()
                f1, ax1 = plt.subplots(1, 1)
                f2, ax2 = plt.subplots(1, 1)

                # usual data
                input_file_name = (
                            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                        usecols=(0))
                    prob_ss_array = data_array.reshape((N, N))
                    prob_ss_x = trapz(prob_ss_array, dx=dx, axis=1)
                    prob_ss_y = trapz(prob_ss_array, dx=dx, axis=0)
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
                ax1.plot(phi_array, prob_ss_x, color='blue', linewidth=1.0)
                ax2.plot(phi_array, prob_ss_y, color='blue', linewidth=1.0)

                # steady state check data
                input_file_name = (
                            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191018_steadystatecheck" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                        usecols=(0))
                    prob_ss_array = data_array.reshape((N, N))
                    prob_ss_x = trapz(prob_ss_array, dx=dx, axis=1)
                    prob_ss_y = trapz(prob_ss_array, dx=dx, axis=0)
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
                ax1.plot(phi_array, prob_ss_x, color='orange', linewidth=1.0, linestyle='--')
                ax2.plot(phi_array, prob_ss_y, color='orange', linewidth=1.0, linestyle='--')

                ax1.set_xlabel('$ \\theta_o $')
                ax1.set_ylabel('$P( \\theta_o) $')
                ax2.set_xlabel('$ \\theta_o $')
                ax2.set_ylabel('$P( \\theta_o) $')
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                f2.savefig(output_file_name2.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                plt.close()


def plot_free_energy_space(target_dir):  # plot of the pmf along some coordinate
    output_file_name1 = (
                target_dir + "FE_force_xy_cond_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 >= abs(psi_2):
                plt.figure()
                f1, ax1 = plt.subplots(2, Ecouple_array.size, figsize=(20, 6), sharey='all', sharex='all')
                force_x = zeros(N)
                force_y = zeros(N)

                force_x = psi_1 * phi_array
                force_y = psi_2 * phi_array

                # force_x = zeros((N,N))
                #                 force_y = zeros((N,N))
                #                 for i in range(N):
                #                     force_x[:,i] = psi_1*phi_array
                #                     force_y[i] = psi_2*phi_array

                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                            usecols=(0, 2))
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        pot_array = data_array[:, 1].reshape((N, N))
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

                    prob_ss_x = trapz(prob_ss_array,
                                      axis=1)  # integrate using axis=1 integrates out the y component, gives us P(x)
                    prob_ss_y = trapz(prob_ss_array, axis=0)
                    prob_ss_x_given_y = prob_ss_array / prob_ss_y[:, None]
                    prob_ss_y_given_x = prob_ss_array / prob_ss_x

                    FE_array_x = trapz(prob_ss_y_given_x * pot_array, axis=1) + trapz(
                        prob_ss_y_given_x * log(prob_ss_y_given_x),
                        axis=1) - force_x - force_y  # - trapz(force_x + force_y, axis=1)
                    # FE_array_x = trapz( prob_ss_y_given_x * (pot_array), axis = 1)
                    # S_array_x = -trapz( prob_ss_y_given_x * log( prob_ss_y_given_x ), axis = 1 )
                    FE_array_x -= amin(FE_array_x)

                    FE_array_y = trapz(prob_ss_x_given_y * pot_array, axis=0) + trapz(
                        prob_ss_x_given_y * log(prob_ss_x_given_y),
                        axis=0) - force_y - force_x  # - trapz(force_x + force_y, axis=0)
                    # FE_array_y = trapz( prob_ss_x_given_y * (pot_array), axis = 0)
                    #                   S_array_y = -trapz( prob_ss_x_given_y * log( prob_ss_x_given_y ), axis = 0 )
                    FE_array_y -= amin(FE_array_y)

                    # ax1[0, ii].plot(phi_array, S_array_x)
                    #                    ax1[1, ii].plot(phi_array, S_array_y)
                    ax1[0, ii].plot(phi_array, FE_array_x)
                    ax1[1, ii].plot(phi_array, FE_array_y)  # pmf(y) = pmf(\theta_1)

                    if (ii == 0):
                        ax1[0, ii].set_title("$E_{couple}$" + "={}".format(Ecouple))
                        ax1[0, ii].set_ylabel('$F( \\theta_\mathrm{o} )$')
                        ax1[1, ii].set_ylabel('$F( \\theta_\mathrm{1} )$')
                    else:
                        ax1[0, ii].set_title("{}".format(Ecouple))
                    ax1[0, ii].set_xlabel('$ \\theta_\mathrm{o} $')
                    ax1[1, ii].set_xlabel('$ \\theta_1 $')
                    ax1[0, ii].spines['right'].set_visible(False)
                    ax1[0, ii].spines['top'].set_visible(False)
                    ax1[1, ii].spines['right'].set_visible(False)
                    ax1[1, ii].spines['top'].set_visible(False)
                    ax1[0, ii].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    ax1[1, ii].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    ax1[1, ii].set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
                    ax1[1, ii].set_xticklabels([0, '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$2 \pi$'])

                f1.tight_layout()
                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                plt.close()


def plot_pmf_space(target_dir):  # plot of the pmf along some coordinate
    output_file_name1 = (
                target_dir + "pmf_x_cond_force_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    output_file_name2 = (
                target_dir + "pmf_y_cond_force_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 >= abs(psi_2):
                plt.figure()
                f1, ax1 = plt.subplots(1, Ecouple_array.size, figsize=(20, 3), sharey='all')
                f2, ax2 = plt.subplots(1, Ecouple_array.size, figsize=(20, 3), sharey='all')

                # force_x = zeros((N,N))
                # force_y = zeros((N,N))
                #
                # for i in range(N):
                #     force_x[i] = psi_1*phi_array
                #     force_y[:,i] = psi_2*phi_array

                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                            usecols=(0, 2))
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        pot_array = data_array[:, 1].reshape((N, N))
                        prob_ss_x = trapz(prob_ss_array,
                                          axis=1)  # integrate using axis=1 integrates out the y component, gives us P(x)
                        prob_ss_y = trapz(prob_ss_array, axis=0)
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

                    prob_ss_x_given_y = prob_ss_array / prob_ss_y[:, None]
                    prob_ss_y_given_x = prob_ss_array / prob_ss_x

                    pmf_array_x = - log(trapz(prob_ss_y_given_x * exp(- pot_array), axis=1))  # pmf(\theta_o) = pmf(x)
                    ax1[ii].plot(phi_array, pmf_array_x)

                    if (ii == 0):
                        ax1[ii].set_title("$E_{couple}$" + "={}".format(Ecouple))
                        ax1[ii].set_ylabel('$PMF( \\theta_\mathrm{o} )$')
                    else:
                        ax1[ii].set_title("{}".format(Ecouple))
                    ax1[ii].set_xlabel('$ \\theta_\mathrm{o} $')
                    ax1[ii].spines['right'].set_visible(False)
                    ax1[ii].spines['top'].set_visible(False)
                    ax1[ii].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    ax1[ii].set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
                    ax1[ii].set_xticklabels([0, '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$2 \pi$'])

                    pmf_array_y = - log(trapz(prob_ss_x_given_y * exp(- pot_array), axis=0))
                    ax2[ii].plot(phi_array, pmf_array_y)  # pmf(y) = pmf(\theta_1)

                    ax2[ii].set_xlabel('$ \\theta_1 $')
                    if ii == 0:
                        ax2[ii].set_title("$E_{couple}$" + "={}".format(Ecouple))
                        ax2[ii].set_ylabel('$PMF( \\theta_1 )$')
                    else:
                        ax2[ii].set_title("{}".format(Ecouple))
                    ax2[ii].spines['right'].set_visible(False)
                    ax2[ii].spines['top'].set_visible(False)
                    ax2[ii].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    ax2[ii].set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
                    ax2[ii].set_xticklabels([0, '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$2 \pi$'])

                f1.tight_layout()
                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

                f2.tight_layout()
                f2.savefig(output_file_name2.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                plt.close()


def plot_pmf_barrier_Ecouple(target_dir):  # plot of the pmf along some coordinate
    output_file_name1 = (
                target_dir + "pmf_x_marg_barrier_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    output_file_name2 = (
                target_dir + "pmf_y_marg_barrier_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:  # different figures
            if psi_1 > abs(psi_2):
                plt.figure()
                f1, ax1 = plt.subplots(1, figsize=(5, 4), sharey='all')
                f2, ax2 = plt.subplots(1, figsize=(5, 4), sharey='all')

                for jj, phi in enumerate(phase_array):  # different lines
                    min_pmf_x = zeros(len(Ecouple_array))
                    max_pmf_x = zeros(len(Ecouple_array))
                    min_pmf_y = zeros(len(Ecouple_array))
                    max_pmf_y = zeros(len(Ecouple_array))

                    for ii, Ecouple in enumerate(Ecouple_array):  # different points in a line
                        input_file_name = (
                                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(
                                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                                usecols=(0, 2))
                            prob_ss_array = data_array[:, 0].reshape((N, N))
                            pot_array = data_array[:, 1].reshape((N, N))
                            prob_ss_x = trapz(prob_ss_array, axis=1)
                            prob_ss_y = trapz(prob_ss_array, axis=0)
                        except OSError:
                            print('Missing file')
                            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

                        prob_ss_x_given_y = prob_ss_array / prob_ss_y
                        prob_ss_y_given_x = prob_ss_array / prob_ss_x
                        pmf_array_y = - log(trapz(prob_ss_x * exp(- pot_array), axis=0))
                        pmf_array_x = - log(trapz(prob_ss_y * exp(- pot_array), axis=1))

                        min_pmf_x[ii] = amin(pmf_array_x)
                        max_pmf_x[ii] = amax(pmf_array_x)
                        min_pmf_y[ii] = amin(pmf_array_y)
                        max_pmf_y[ii] = amax(pmf_array_y)

                    barrier_height_x = max_pmf_x - min_pmf_x
                    barrier_height_y = max_pmf_y - min_pmf_y

                    ax1.plot(Ecouple_array, barrier_height_x, '-', label=label_lst[jj],
                             color=plt.cm.cool(colorlist[jj]))
                    ax2.plot(Ecouple_array, barrier_height_y, '-', label=label_lst[jj],
                             color=plt.cm.cool(colorlist[jj]))

                ax1.set_ylabel('$ E^{\u2021}_\mathrm{pmf,x} $')
                ax1.set_xlabel('$ E_\mathrm{couple} $')
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax1.set_xscale('log')
                ax1.legend(loc='best')

                ax2.set_ylabel('$ E^{\u2021}_\mathrm{pmf,y} $')
                ax2.set_xlabel('$ E_\mathrm{couple} $')
                ax2.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax2.set_xscale('log')
                ax2.legend(loc='best')

                f1.tight_layout()
                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

                f2.tight_layout()
                f2.savefig(output_file_name2.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

                plt.close()


def plot_scatter_pmf_power(target_dir):
    output_file_name1 = (
                target_dir + "scatterplot_FE_cond_x_power_extra_force_tight_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    output_file_name2 = (
                target_dir + "scatterplot_FE_cond_y_power_extra_force_tight_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:  # different figures
            if psi_1 >= abs(psi_2):
                plt.figure()
                f1, ax1 = plt.subplots(1, figsize=(5, 4), sharey='all')
                f2, ax2 = plt.subplots(1, figsize=(5, 4), sharey='all')
                ax1.plot()

                force_x = zeros(N)
                force_y = zeros(N)

                force_x = psi_1 * phi_array
                force_y = psi_2 * phi_array

                # force_x = zeros((N,N))
                #                 force_y = zeros((N,N))
                #
                #                 for i in range(N):
                #                     force_x[:,i] = psi_1*phi_array
                #                     force_y[i] = psi_2*phi_array

                ## calculate barrier height
                barrier_height_x = zeros((len(Ecouple_tot_array), len(phase_array)))
                barrier_height_y = zeros((len(Ecouple_tot_array), len(phase_array)))
                # plt1 = zeros(len(phase_array))
                for jj, phi in enumerate(phase_array):
                    min_pmf_x = zeros(len(Ecouple_tot_array))
                    max_pmf_x = zeros(len(Ecouple_tot_array))
                    min_pmf_y = zeros(len(Ecouple_tot_array))
                    max_pmf_y = zeros(len(Ecouple_tot_array))

                    for ii, Ecouple in enumerate(Ecouple_tot_array):
                        if Ecouple in Ecouple_array:
                            input_file_name = (
                                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                        else:
                            input_file_name = (
                                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190610_phaseoffset_extra" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(
                                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                                usecols=(0, 2))
                            prob_ss_array = data_array[:, 0].reshape((N, N))
                            pot_array = data_array[:, 1].reshape((N, N))
                            prob_ss_x = trapz(prob_ss_array, axis=1)
                            prob_ss_y = trapz(prob_ss_array, axis=0)
                        except OSError:
                            print('Missing file')
                            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

                        prob_ss_x_given_y = prob_ss_array / prob_ss_y[:, None]
                        prob_ss_y_given_x = prob_ss_array / prob_ss_x
                        pmf_array_x = trapz(prob_ss_y_given_x * (pot_array + log(prob_ss_y_given_x)),
                                            axis=1) - force_x - force_y
                        pmf_array_y = trapz(prob_ss_x_given_y * (pot_array + log(prob_ss_x_given_y)),
                                            axis=0) - force_x - force_y

                        # mins_x = (diff(sign(diff(pmf_array_x))) > 0).nonzero()[0] #positions of minima
                        # maxs_x = (diff(sign(diff(pmf_array_x))) < 0).nonzero()[0]
                        # mins_y = (diff(sign(diff(pmf_array_y))) > 0).nonzero()[0]
                        # maxs_y = (diff(sign(diff(pmf_array_y))) < 0).nonzero()[0]
                        #
                        # if mins_x[0] > 100 or len(mins_x) < 3:
                        #     try:
                        #         barrier_height_x[ii, jj] = pmf_array_x[mins_x[0]] - pmf_array_x[mins_x[1]]
                        #     except:
                        #         barrier_height_x[ii, jj] = float('NaN')
                        # else:
                        #     try:
                        #         barrier_height_x[ii, jj] = pmf_array_x[mins_x[1]] - pmf_array_x[mins_x[2]]
                        #     except:
                        #         barrier_height_x[ii, jj] = float('NaN')
                        #
                        # if mins_y[0] > 100 or len(mins_y) < 3:
                        #     try:
                        #         barrier_height_y[ii, jj] = pmf_array_y[mins_y[0]] - pmf_array_y[mins_y[1]]
                        #     except:
                        #         barrier_height_y[ii, jj] = float('NaN')
                        # else:
                        #     try:
                        #         barrier_height_y[ii, jj] = pmf_array_y[mins_y[1]] - pmf_array_y[mins_y[2]]
                        #     except:
                        #         barrier_height_y[ii, jj] = float('NaN')
                        # print(mins_y)

                        min_pmf_x[ii] = amin(pmf_array_x)
                        max_pmf_x[ii] = amax(pmf_array_x)
                        min_pmf_y[ii] = amin(pmf_array_y)
                        max_pmf_y[ii] = amax(pmf_array_y)

                        barrier_height_x[ii, jj] = max_pmf_x[ii] - min_pmf_x[ii]
                        barrier_height_y[ii, jj] = max_pmf_y[ii] - min_pmf_y[ii]

                ## grab power data
                power_array = zeros((len(Ecouple_tot_array), len(phase_array)))
                for ii, Ecouple in enumerate(Ecouple_tot_array):
                    if Ecouple in Ecouple_array:
                        input_file_name = (
                                    target_dir + "190624_Twopisweep_complete_set/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    else:
                        input_file_name = (
                                    target_dir + "190610_Extra_measurements_Ecouple/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                            usecols=(4))
                        power_array[ii, :] = data_array[:len(phase_array)]
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                ## plot correlation
                for jj, phi in enumerate(phase_array):
                    ax1.plot(power_array[:, jj], barrier_height_x[:, jj], linestyle='-',
                             color=plt.cm.cool(colorlist[jj]), marker='.', label=label_lst[jj])  #
                    ax2.plot(power_array[:, jj], barrier_height_y[:, jj], linestyle='-',
                             color=plt.cm.cool(colorlist[jj]), marker='.', label=label_lst[jj])
                ax1.set_xlabel('$ P_\mathrm{ATP/ADP} $')
                ax1.set_ylabel('$ E^{\u2021}_\mathrm{F,x} $')
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax1.set_xlim([-0.0004, 0])
                ax1.set_ylim([0, None])
                ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                ax1.legend(loc='best')

                f1.tight_layout()
                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

                ax2.set_xlabel('$ P_\mathrm{ATP/ADP} $')
                ax2.set_ylabel('$ E^{\u2021}_\mathrm{F,y} $')
                ax2.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.set_xlim([-0.0004, 0])
                ax2.set_ylim([0, None])
                ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                ax2.legend(loc='best')

                f2.tight_layout()
                f2.savefig(output_file_name2.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                plt.close()


def plot_prob_coord(target_dir):  # plot of the pmf along some coordinate
    output_file_name1 = (
                target_dir + "prob_space_plot_scaled_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 >= abs(psi_2):
                plt.figure()
                f1, ax1 = plt.subplots(1, Ecouple_array.size, figsize=(18, 3), sharey='all')

                ##Find max prob. to set plot range
                input_file_name = (
                            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, 128.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=(0))
                    prob_ss_array = data_array.reshape((N, N))
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

                prob_max = amax(prob_ss_array)

                ##plots
                for ii, Ecouple in enumerate(Ecouple_array):
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                            usecols=(0, 2))
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        pot_array = data_array[:, 1].reshape((N, N))
                        prob_ss_x = trapz(prob_ss_array,
                                          axis=1)  # integrate using axis=1 integrates out the y component, gives us P(x)
                        prob_ss_y = trapz(prob_ss_array, axis=0)
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

                    prob_ss_new = zeros((N, N))

                    for i in range(N):
                        for j in range(N):
                            # prob_ss_new[int(floor(0.5*(i + j))), int(floor(0.5*(i - j)) + 180)] = prob_ss_array[i,j]
                            prob_ss_new[i, (j + 180) % N] = prob_ss_array[(i + j) % N, (i - j) % N]

                    ax1[ii].contourf(prob_ss_new[:, 90:271], vmin=0, vmax=prob_max)

                    if (ii == 0):
                        ax1[ii].set_title("$E_{couple}$" + "={}".format(Ecouple))
                        ax1[ii].set_ylabel('$\\theta_\mathrm{cm}$')
                    else:
                        ax1[ii].set_title("{}".format(Ecouple))
                    ax1[ii].set_xlabel('$\\theta_\mathrm{diff}$')
                    ax1[ii].spines['right'].set_visible(False)
                    ax1[ii].spines['top'].set_visible(False)
                    ax1[ii].set_xticks([0, 45, 90, 135, 180])
                    ax1[ii].set_xticklabels(['$-\pi$', '', '0', '', '$ \pi$'])
                    ax1[ii].set_yticks([0, 60, 120, 180, 240, 300, 360])
                    ax1[ii].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])

                f1.tight_layout()
                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

                plt.close()


def plot_pmf_coord(target_dir):  # plot of the pmf along some coordinate
    output_file_name1 = (
                target_dir + "pmf_coord1_cond_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    output_file_name2 = (
                target_dir + "pmf_coord2_cond_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 >= abs(psi_2):
                plt.figure()
                f1, ax1 = plt.subplots(1, Ecouple_array.size, figsize=(20, 3), sharey='all')
                f2, ax2 = plt.subplots(1, Ecouple_array.size, figsize=(20, 3), sharey='all')

                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                            usecols=(0, 2))
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        pot_array = data_array[:, 1].reshape((N, N))
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

                    prob_ss_new = zeros((N, N))
                    pot_new = zeros((N, N))

                    for i in range(N):
                        for j in range(N):
                            prob_ss_new[i, (j + 180) % N] = prob_ss_array[(i + j) % N, (i - j) % N]
                            pot_new[i, (j + 180) % N] = pot_array[(i + j) % N, (i - j) % N]

                    prob_ss_1 = trapz(prob_ss_new, axis=1)
                    prob_ss_2 = trapz(prob_ss_new, axis=0)
                    prob_ss_1_given_2 = prob_ss_new / prob_ss_2
                    prob_ss_2_given_1 = prob_ss_new / prob_ss_1

                    pmf_array_1 = - log(trapz(prob_ss_2_given_1 * exp(- pot_new), axis=1))
                    ax1[ii].plot(phi_array, pmf_array_1)

                    if (ii == 0):
                        ax1[ii].set_title("$E_{couple}$" + "={}".format(Ecouple))
                        ax1[ii].set_ylabel('$PMF( \\theta_\mathrm{cm} )$')
                    else:
                        ax1[ii].set_title("{}".format(Ecouple))
                    ax1[ii].set_xlabel('$ \\theta_\mathrm{cm} $')
                    ax1[ii].spines['right'].set_visible(False)
                    ax1[ii].spines['top'].set_visible(False)
                    ax1[ii].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    ax1[ii].set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
                    ax1[ii].set_xticklabels([0, '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$2 \pi$'])

                    pmf_array_2 = - log(trapz(prob_ss_1_given_2 * exp(- pot_new), axis=0))
                    ax2[ii].plot(phi_array, pmf_array_2)

                    ax2[ii].set_xlabel('$ \\theta_\mathrm{diff} $')
                    if ii == 0:
                        ax2[ii].set_title("$E_{couple}$" + "={}".format(Ecouple))
                        ax2[ii].set_ylabel('$PMF( \\theta_\mathrm{diff} )$')
                    else:
                        ax2[ii].set_title("{}".format(Ecouple))
                    ax2[ii].spines['right'].set_visible(False)
                    ax2[ii].spines['top'].set_visible(False)
                    ax2[ii].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    ax2[ii].set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
                    ax2[ii].set_xticklabels(['$-2 \pi$', '', '$-2 \pi/3$', '', '$2 \pi/3$', '', '$2 \pi$'])

                f1.tight_layout()
                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

                f2.tight_layout()
                f2.savefig(output_file_name2.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                plt.close()


def calculate_lag(target_dir):
    psi_1 = 4.0
    psi_2 = -2.0
    # phase_offset = 0.0
    phi_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944])
    # phi_array = array([0.0])

    lag_data = zeros((phase_array.size, Ecouple_array.size))
    super_power = zeros((phase_array.size, Ecouple_array.size))
    max_lag = zeros(Ecouple_array.size)
    max_phi = zeros(Ecouple_array.size)

    # calculating the lag from probability distributions
    for i, Ecouple in enumerate(Ecouple_array):
        for j, phase_offset in enumerate(phase_array):

            if phase_offset in phi_array:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" +
                                   "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            else:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints" +
                                   "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")

            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_offset),
                    usecols=0)
                prob_ss_array = data_array.reshape((N, N))
                # prob_ss_array = array(prob_ss_array[:120, :120])
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_offset))

            # mode lag
            # maxprob = amax(prob_ss_array)
            # pos = where(prob_ss_array == maxprob)
            # # print(maxprob, pos)
            # # print(Ecouple, ((pos[0]-pos[1]) % 120))
            # lag_data[j, i] = array((pos[0]-pos[1]) % 120)

            # mean lag
            # define moving window to calculate lag accurately
            angle = zeros((N, N))
            for ii in range(N):
                if ii <= int(N / 2):
                    angle[ii, :(ii + int(N / 2))] = array(linspace(0, ii * dx + pi, ii + int(N / 2), endpoint=False))
                    angle[ii, (ii + int(N / 2)):] = array(
                        linspace(ii * dx - pi, 2 * pi, int(N / 2) - ii, endpoint=False))
                else:
                    angle[ii, :(ii - int(N / 2))] = array(
                        linspace(2 * pi, ii * dx + pi, ii - int(N / 2), endpoint=False))
                    angle[ii, (ii - int(N / 2)):] = array(
                        linspace(ii * dx - pi, 2 * pi, int(3 * N / 2) - ii, endpoint=False))

            Pss_window = prob_ss_array * angle
            for ii in range(N):  # loop over Fo positions
                if ii < int(N / 4):
                    # Pss_window[ii, :] =
                    Pss_window[ii, (int(N / 4) + ii):(int(3 * N / 4) + ii)] = 0
                elif ii > int(3 * N / 4):
                    Pss_window[ii, (ii - int(3 * N / 2)):(ii - int(N / 4))] = 0
                else:
                    Pss_window[ii, :(ii - int(N / 4))] = 0
                    Pss_window[ii, (ii + int(N / 4)):] = 0

            av_prob_x = trapz(trapz(Pss_window.T * angle, dx=1, axis=1), dx=1, axis=0)
            av_prob_y = trapz(trapz(Pss_window * angle, dx=1, axis=1), dx=1, axis=0)
            # print(Ecouple, av_prob_x, av_prob_y)
            lag_data[j, i] = av_prob_x - av_prob_y

        # print(amax(lag_data[:, i]))
        # print(where(lag_data[:, i] == amax(lag_data[:, i]))[0])
        # if len(where(lag_data[:, i] == amax(lag_data[:, i]))[0]) == 1:
        #     max_lag[i] = where(lag_data[:, i] == amax(lag_data[:, i]))[0]
        # else:
        #     # print(where(lag_data[:, i] == amax(lag_data[:, i]))[0])
        #     max_lag[i] = where(lag_data[:, i] == amax(lag_data[:, i]))[0][0]

    print(lag_data)

    # Calculate the phase offset that leads to the highest power output
    for i, Ecouple in enumerate(Ecouple_array):
        input_file_name = (target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
                           + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            phi_array = data_array[:, 0]
            power_y_array = -data_array[:, 1]
            super_power[:, i] = power_y_array
        except OSError:
            print('Missing file')

        # print(where(power_y_array == amax(power_y_array))[0][0])
        # max_phi[i] = where(power_y_array == amax(power_y_array))[0][0]

    # print(max_phi)
    # print(super_power)

    plt.figure()
    f1, ax1 = plt.subplots(1, 1, figsize=(6, 6), sharey='all')
    ax1.axhline(0, color='grey', linewidth=1, linestyle='--', label='_nolegend_')
    ax1.axvline(0, color='grey', linewidth=1, linestyle='--', label='_nolegend_')
    ax1.plot(super_power, lag_data / (2 * pi), marker='o', linestyle='-')
    # ax1.set_xlim((0, 11))
    # ax1.set_ylim((0, 11))
    ax1.set_xlabel('$\\beta \mathcal{P}_{\\rm ATP}\ (t_{\\rm sim}^{-1})$', fontsize=20)
    ax1.set_ylabel('$\phi_{\\rm Lag}\ (\\rm rev)$', fontsize=20)
    # ax1.set_xticks(range(0, 2*len(max_phi), 2))
    # ax1.set_xticklabels(['$0$', '', '', '$1/6$', '', '', '$1/3$'])
    # ax1.set_yticks(range(0, 2*len(max_phi), 2))
    # ax1.set_yticklabels(['$0$', '', '', '$1/6$', '', '', '$1/3$'])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(Ecouple_array)

    f1.tight_layout()
    output_file_name1 = (
            target_dir + "Lag_power_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_path(target_dir):
    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" +
            "/reference_" +
            "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
            "_outfile.dat")
    Ecouple_array = [8.0, 16.0, 32.0]
    psi_1 = 4.0
    psi_2 = -2.0
    angle = array(linspace(0, 2 * pi, N, endpoint=False))

    plt.figure()
    f1, ax1 = plt.subplots(1, 3, figsize=(12, 4), sharey='all')
    flux_X = zeros(N)
    flux_Y = zeros(N)
    traj_X = zeros(100000)
    traj_Y = zeros(100000)

    for i, Ecouple in enumerate(Ecouple_array):
        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, 3.0, 3.0, 0.0))
        prob_ss_array = data_array[:, 0].reshape((N, N))
        prob_eq_array = data_array[:, 1].reshape((N, N))
        pot_array = data_array[:, 2].reshape((N, N))
        drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
        diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))

        flux_array = zeros((2, N, N))
        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
        flux_array = asarray(flux_array) / (dx * dx)

        flux_X = (1. / (2 * pi)) * trapz(flux_array[0, ...], dx=dx, axis=0)
        flux_Y = (1. / (2 * pi)) * trapz(flux_array[1, ...], dx=dx, axis=1)

        current_X = 0
        dt = 0.3
        j = 1
        k = 0
        while current_X < 2 * pi and j < 100000:
            current_X += 2 * pi * flux_X[k] * dt
            traj_X[j] += current_X
            j += 1
            if current_X > positions[k + 1]:
                k += 1

        # prob_ss_x = trapz(prob_ss_array, dx=1, axis=1)
        # prob_ss_y = trapz(prob_ss_array, dx=1, axis=0)
        #
        # for j in range(N):
        #     prob_y_given_x[j, :] = prob_ss_array[j, :]/prob_ss_x[j]
        #     prob_x_given_y[j, :] = prob_ss_array[j, :]/prob_ss_y[j]

        # test = trapz(prob_y_given_x, dx=1, axis=0)

        ax1[i].contourf(angle, angle, flux_array[0, ...])
        # ax1[i].plot(linspace(0, traj_X[-1], 100000), traj_X, linestyle='-')
        # ax1[i].legend()
        # ax1[i].set_ylim((0, 0.05))

    f1.tight_layout()
    output_filename = (
            target_dir + "PSS_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    f1.savefig(output_filename.format(E0, E1, psi_1, psi_2, 3.0, 3.0))


def plot_power_Ecouple_grid_extended(target_dir):  # grid of plots of the flux as a function of the phase offset

    output_file_name = (target_dir + "power_ATP_Ecouple_grid_extended_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 6, sharex='col', sharey='row', figsize=(16, 6))
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = round(-psi_1 / ratio, 2)
            print(psi_1, psi_2)

            # line at infinite Ecouple power
            input_file_name = (
                    target_dir + "200220_moregrid/processed_data/"
                    + "Power_Ecouple_inf_grid_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_n1_3.0_n2_3.0_outfile.dat")
            try:
                inf_array = loadtxt(input_file_name, usecols=2)
                axarr[i, j].axhline(inf_array[i * 6 + j], color='grey', linestyle=':', linewidth=1)
            except OSError:
                print('Missing file Infinite Power Coupling')

            # zero-barrier result
            if ratio == 8 or ratio == 4 or ratio == 2:
                input_file_name = (
                        target_dir + "191217_morepoints/processed_data/"
                        + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            else:
                input_file_name = (
                        target_dir + "200220_moregrid/processed_data/"
                        + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                Ecouple_array2 = array(data_array[:, 0])
                flux_y_array = array(data_array[:, 2])
                power_y = -flux_y_array * psi_2
                axarr[i, j].plot(Ecouple_array2, power_y, '-', color='C0', linewidth=3)
            except OSError:
                print('Zero-barrier file missing')

            # E0=E1=2 barrier data
            power_y_array = []
            if ratio == 8 or ratio == 4 or ratio == 2:
                Ecouple_array_tot = array(
                    [2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
            else:
                Ecouple_array_tot = array(
                    [8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                if ratio == 8 or ratio == 4 or ratio == 2:
                    input_file_name = (
                            target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                else:
                    input_file_name = (
                            target_dir + "200220_moregrid/processed_data/" + "flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=4)

                    if data_array.size > 2:
                        power_y = array(data_array[0])
                    else:
                        power_y = array(data_array)
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file flux')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

            axarr[i, j].plot(Ecouple_array_tot, -power_y_array, '.', color='C1', markersize=14, marker='.')

            print('Max power/ infinite power', amax(-power_y_array) / inf_array[i * 6 + j])

            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            # axarr[i, j].spines['bottom'].set_visible(False)
            axarr[i, j].tick_params(axis='both', labelsize=14)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((2, 150))
            elif j == 1:
                axarr[i, j].set_xlim((3, 150))
            elif j == 2:
                axarr[i, j].set_xlim((5, 150))
            elif j == 3:
                axarr[i, j].set_xlim((6, 150))
            elif j == 4:
                axarr[i, j].set_xlim((7, 150))
            else:
                axarr[i, j].set_xlim((8.5, 150))

            if i == 0:
                axarr[i, j].set_ylim((0, 0.000085))
                axarr[i, j].set_yticks([0, 0.00004, 0.00008])
                axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            elif i == 1:
                axarr[i, j].set_ylim((0, 0.00033))
                axarr[i, j].set_yticks([0, 0.00015, 0.0003])
                axarr[i, j].set_yticklabels([r'$0$', r'$15$', r'$30$'])
            else:
                axarr[i, j].set_ylim((0, 0.0013))
                axarr[i, j].set_yticks([0, 0.0005, 0.001])
                axarr[i, j].set_yticklabels([r'$0$', r'$50$', r'$100$'])

            if j == 0 and i > 0:
                axarr[i, j].yaxis.offsetText.set_fontsize(0)
            else:
                axarr[i, j].yaxis.offsetText.set_fontsize(14)

            if j == psi_ratio.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi1_array[i], labelpad=16, rotation=270, fontsize=14)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.2f$' % psi_ratio[j], fontsize=14)

    f.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=20)
    f.text(0.01, 0.5, r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', va='center', rotation='vertical',
           fontsize=20)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=20)
    # f.text(0.5, 0.95, r'$2 \pi \beta \mu_{\rm H^+}\ (\rm rev^{-1})$', ha='center', fontsize=20)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (\rm k_{\rm B} T / rad)$', va='center', rotation=270, fontsize=20)
    # f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))

def plot_power_ratio_Ecouple_grid(target_dir):
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 22.63, 24.0,
         32.0, 45.25, 64.0, 90.51, 128.0])
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    colorlst = ['C1', 'C2', 'C3']
    barrier_heights = array([2.0, 4.0])

    output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/results/FP_Full_2D/" +
                        "P_ATP_P_inf_log_Ecouple_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")

    f, axarr = plt.subplots(3, 3, sharex='col', sharey='all', figsize=(8, 6))

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print('Chemical driving forces:', psi_1, psi_2)

            axarr[i, j].axhline(1, color='grey', linestyle='--', linewidth=2)

            # Fokker-Planck results (2 kT barriers)
            for k, E0 in enumerate(barrier_heights):
                E1 = E0

                # Power at infinite coupling
                if E0 == 2.0:
                    input_file_name = (
                                "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/"
                                + "Power_Ecouple_inf_grid_E0_2.0_E1_2.0_n1_3.0_n2_3.0_outfile.dat")
                else:
                    input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/Rigid_coupling/"
                                       + "Power_Ecouple_inf_grid_E0_4.0_E1_4.0_n1_3.0_n2_3.0_outfile.dat")

                try:
                    inf_array = loadtxt(input_file_name, usecols=2)
                except OSError:
                    print('Missing file Infinite Power Coupling')

                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array_tot):
                    if Ecouple in Ecouple_extra and E0 == 2.0:
                        input_file_name = (
                                target_dir + "200511_2kT_extra/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    elif E0 == 4.0:
                        input_file_name = (
                                target_dir + "200506_4kTbarrier/spectral/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    elif E0 == 6.0:
                        input_file_name = (
                                target_dir + "200511_6kTbarrier/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    else:
                        input_file_name = (
                                target_dir + "plotting_data/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        if Ecouple == 22.63 and (E0 == 4.0 or E0 == 6.0):
                            data_array = loadtxt(
                                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, 22.62),
                                usecols=4)
                        else:
                            data_array = loadtxt(
                                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                                usecols=4)

                        if data_array.size > 2:  # data format varies a little
                            power_y = array(data_array[0])
                        else:
                            power_y = array(data_array)
                        power_y_array = append(power_y_array, power_y)
                    except OSError:
                        print('Missing file flux')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                if E0 == 2.0:
                    print(-power_y_array / inf_array[3 * i + j])
                axarr[i, j].plot(Ecouple_array_tot, -power_y_array / inf_array[3 * i + j], '.', color=colorlst[k],
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
                axarr[i, j].set_xlim((5, 150))

            axarr[i, j].set_ylim((0.5, 15))
            axarr[i, j].set_yticks([1, 10])

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
                print( )
                # axarr[i, j].tick_params(axis='x', colors='red', which='both')
                # axarr[i, j].tick_params(axis='y', colors='red', which='both')
                # axarr[i, j].spines['left'].set_color('red')
                # axarr[i, j].spines['bottom'].set_color('red')
            else:
                axarr[i, j].tick_params(axis='both', labelsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=24)
    f.text(0.01, 0.5, r'$\mathcal{P}_{\rm ATP} / \mathcal{P}^{\infty}_{\rm ATP}$', va='center', rotation='vertical',
           fontsize=24)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=24)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (k_{\rm B} T / \rm rad)$', va='center', rotation=270, fontsize=24)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))

def plot_Pss(target_dir):
    output_file_name1 = (
                "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/results/FP_Full_2D/" +
                "Diff_Pss_Peq_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                "_.pdf")

    # psi1_array = array([2.0, 4.0, 8.0])
    # psi2_array = array([-0.25, -0.5, -1.0, -2.0, -4.0])
    psi1_array = array([0.0])
    psi2_array = array([0.0])

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            for ii, Ecouple in enumerate(Ecouple_array):
                plt.figure()
                f1, ax1 = plt.subplots(1, 1)

                # usual data
                input_file_name1 = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" +
                                    "/reference_" +
                                    "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                    "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name1.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                        usecols=(0, 1))
                    N = int(sqrt(len(data_array)))  # check grid size
                    print('Grid size: ', N)
                    prob_ss_array = data_array[:, 0].reshape((N, N))
                    prob_eq_array = data_array[:, 1].reshape((N, N))
                    prob_ss_x = trapz(prob_ss_array, dx=dx, axis=1)
                    prob_ss_y = trapz(prob_ss_array, dx=dx, axis=0)
                    prob_eq_x = trapz(prob_eq_array, dx=dx, axis=1)
                    prob_eq_y = trapz(prob_eq_array, dx=dx, axis=0)
                except OSError:
                    print('Missing file')
                    print(input_file_name1.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

                ax1.plot(positions, prob_ss_x - prob_eq_x, color='C0', linewidth=2.0)
                ax1.plot(positions, prob_ss_y - prob_eq_y, color='C1', linewidth=2.0)

                print(max(abs(prob_ss_array - prob_eq_array)))

                ax1.set_xlabel(r'$ \theta_{\rm o, 1} $')
                ax1.set_ylabel(r'$P_{\rm ss}( \theta_{\rm o, 1}) - P_{\rm eq}( \theta_{\rm o, 1})  $')
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))



if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/" + \
                 "working_directory_cython/"
    # flux_power_efficiency(target_dir)
    # flux_power_efficiency_extrapoints(target_dir)
    # plot_power_phi_grid(target_dir)
    # plot_power_phi_single(target_dir)
    # plot_power_efficiency_phi_single(target_dir)
    # plot_power_Ecouple_grid(target_dir)
    # plot_efficiency_phi_single(target_dir)
    # plot_efficiency_Ecouple_single(target_dir)
    # plot_efficiency_Ecouple_grid(target_dir)
    # plot_flux_grid(target_dir)
    # plot_flux_phi_single(target_dir)
    # plot_flux_Ecouple_single(target_dir)
    # plot_flux_Ecouple_grid(target_dir)
    # plot_flux_contour(target_dir)
    # plot_flux_space(target_dir)
    # plot_power_efficiency_Ecouple_single(target_dir)
    plot_power_Ecouple_single(target_dir)
    # plot_power_Ecouple_scaled(target_dir)
    # plot_energy_flux(target_dir)
    # plot_energy_flux_grid(target_dir)
    # plot_prob_flux(target_dir)
    # plot_prob_flux_grid(target_dir)
    # plot_condprob_grid(target_dir)
    # plot_rel_flux_Ecouple(target_dir)
    # plot_marg_prob_space(target_dir)
    # plot_free_energy_space(target_dir)
    # plot_pmf_space(target_dir)
    # plot_pmf_barrier_Ecouple(target_dir)
    # plot_scatter_pmf_power(target_dir)
    # plot_prob_coord(target_dir)
    # plot_pmf_coord(target_dir)
    # calculate_lag(target_dir)
    # plot_path(target_dir)
    # plot_power_ratio_Ecouple_grid(target_dir)
    # plot_Pss(target_dir)
