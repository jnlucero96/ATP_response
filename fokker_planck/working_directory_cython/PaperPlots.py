from numpy import array, linspace, loadtxt, append, pi, delete, amax
import math
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

N = 360
dx = 2 * math.pi / N
positions = linspace(0, 2 * math.pi - dx, N)
E0 = 2.0
E1 = 2.0
num_minima1 = 3.0
num_minima2 = 3.0

psi1_array = array([4.0])
psi2_array = array([-2.0])
# psi1_array = array([2.0, 4.0, 8.0])
# psi2_array = array([-0.25, -0.5, -1.0, -2.0, -4.0])
# psi_ratio = array([8, 4, 2])

Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]) #twopisweep
Ecouple_array_extra = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0])  # extra measurements
Ecouple_array_extra2 = array([1.41, 2.83, 5.66, 11.31, 22.63, 45.25, 90.51])
Ecouple_array_tot = array([1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
# Ecouple_array_tot = array(
#     [1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 22.63, 24.0, 32.0, 45.25, 64.0,
#      90.51, 128.0])  # fig 1

# phase_array = array([0.0, 0.175, 0.349066, 0.524, 0.698132, 0.873, 1.0472, 1.222, 1.39626, 1.571, 1.74533, 1.92,
#                      2.0944])  # selection of twopisweep
# phase_array_test = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944])

min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])
color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']

def plot_power_efficiency_Ecouple(target_dir):  # plot power and efficiency vs coupling strength
    output_file_name = (
            target_dir + "power_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 8))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # power plot
            axarr[0].axhline(0, color='black', linewidth=0.5)  # line at zero
            maxpower = 0.000085247
            axarr[0].axhline(maxpower, color='grey', linestyle=':', linewidth=1)  # line at infinite power coupling
            axarr[0].axvline(12, color='grey', linestyle=':', linewidth=1)  # lining up features in the two plots

            # zero-barrier theory lines
            input_file_name = (target_dir + "190624_Twopisweep_complete_set/processed_data/"
                               + "Flux_zerobarrier_evenlyspaced_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2
            axarr[0].plot(Ecouple_array2, power_y, '-', color='C0', label='$0$', linewidth=2)

            # General data
            i = 0  # only use phase=0 data
            power_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 4))
                    if Ecouple in Ecouple_array:
                        power_y = array(data_array[i, 1])
                    else:
                        power_y = array(data_array[1])
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file flux')
            axarr[0].plot(Ecouple_array_tot, -power_y_array, 'o', color='C1', label='$2$', markersize=8)

            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0].yaxis.offsetText.set_fontsize(14)
            axarr[0].tick_params(axis='y', labelsize=14)
            axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (t_{\rm sim}^{-1}) $', fontsize=20)
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            axarr[0].spines['bottom'].set_visible(False)
            axarr[0].set_xlim((1.7, 135))
            axarr[0].set_ylim((-0.0006, 0.00035))

            leg = axarr[0].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)

            # efficiency plot
            axarr[1].axhline(0, color='black', linewidth=1)
            axarr[1].axvline(12, color='grey', linestyle=':', linewidth=1)
            axarr[1].axhline(1, color='grey', linestyle=':', linewidth=1)
            # zero-barrier curve
            input_file_name = (
                    target_dir + "190624_Twopisweep_complete_set/processed_data/"
                    + "Flux_zerobarrier_evenlyspaced_psi1_{0}_psi2_{1}_outfile.dat")
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

            # General data
            eff_array = []
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (
                        target_dir + "191217_morepoints/processed_data/flux_power_efficiency_"
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
            axarr[1].spines['right'].set_visible(False)
            axarr[1].spines['top'].set_visible(False)
            axarr[1].spines['bottom'].set_visible(False)
            axarr[1].set_yticks([0, 0.5, 1.0])
            axarr[1].tick_params(axis='both', labelsize=14)

            f.text(0.05, 0.95, r'$\mathbf{a)}$', ha='center', fontsize=20)
            f.text(0.05, 0.48, r'$\mathbf{b)}$', ha='center', fontsize=20)
            f.subplots_adjust(hspace=0.01)
            f.tight_layout()
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_power_Ecouple_grid(target_dir):  # grid of plots of the flux as a function of the phase offset
    output_file_name = (target_dir + "power_ATP_Ecouple_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='row', figsize=(8, 6))
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print(psi_1, psi_2)

            # line at highest Ecouple power
            input_file_name = (
                            target_dir + "190624_Twopisweep_complete_set/processed_data/"
                            + "Power_Ecouple_inf_grid_E0_2.0_E1_2.0_n1_3.0_n2_3.0_outfile.dat")
            try:
                inf_array = loadtxt(input_file_name, usecols=2)
            except OSError:
                print('Missing file Infinite Power Coupling')

            axarr[i, j].axhline(inf_array[i*3 + j], color='grey', linestyle=':', linewidth=1)
            # input_file_name = (
            #             target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
            #             + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            # try:
            #     data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, 128.0),
            #                          usecols=4)
            #     if data_array.size > 2:
            #         power_y = array(data_array[0])
            #     else:
            #         power_y = array(data_array)
            #     axarr[i, j].axhline(power_y, color='grey', linestyle=':', linewidth=1)
            # except OSError:
            #     print('Missing file flux')

            # line at zero power
            # axarr[i, j].axhline(0, color='black', linewidth=1)

            # zero-barrier result
            input_file_name = (
                        target_dir + "191217_morepoints/processed_data/"
                        + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2

            axarr[i, j].plot(Ecouple_array2, power_y, '-', color='C0', linewidth=3)

            # E0=E1=2 barrier data
            power_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (
                            target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
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
            axarr[i, j].plot(Ecouple_array_tot, -power_y_array, '.', color='C1', markersize=14)

            print('Max power/ infinite power', amax(-power_y_array)/inf_array[i*3 + j])

            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            # axarr[i, j].spines['bottom'].set_visible(False)
            axarr[i, j].tick_params(axis='both', labelsize=14)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((1.6, 150))
            elif j == 1:
                axarr[i, j].set_xlim((2.3, 150))
            else:
                axarr[i, j].set_xlim((5, 150))

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

def plot_power_efficiency_phi(target_dir): # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "power_efficiency_phi_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 6))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # flux plot
            axarr[0].axhline(0, color='black', linewidth=1)  # line at zero

            # zero-barrier theory lines
            input_file_name = (
                        target_dir + "190624_Twopisweep_complete_set/processed_data/"
                        + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2
            axarr[0].axhline(power_y[17], color='C0', linewidth=2, label='$0$')

            # General data
            for ii, Ecouple in enumerate([16.0]):
                input_file_name = (
                            target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 1))
                    phase_array = array(data_array[:, 0])
                    flux_y_array = array(data_array[:, 1])
                except OSError:
                    print('Missing file flux')
            power_y = -flux_y_array * psi_2
            axarr[0].plot(phase_array, power_y, 'o', color='C1', label='$2$', markersize=8)

            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0].yaxis.offsetText.set_fontsize(14)
            axarr[0].tick_params(axis='both', labelsize=14)
            axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', fontsize=20)
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            axarr[0].set_ylim((0, None))
            axarr[0].set_xlim((0, 2.1))

            # efficiency plot
            axarr[1].axhline(0, color='black', linewidth=1)
            axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')
            axarr[1].set_aspect(0.5)

            for ii, Ecouple in enumerate([16.0]):
                input_file_name = (
                            target_dir + "191217_morepoints/processed_data/flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
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
            axarr[1].set_yticks([0, 0.5, 1.0])
            axarr[1].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
            axarr[1].set_xticklabels(['$0$', '', '', '$1/6$', '', '', '$1/3$'])

            leg = axarr[1].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)

            f.text(0.55, 0.07, r'$\phi\ (\rm rev)$', fontsize=20, ha='center')
            f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
            f.text(0.03, 0.37, r'$\mathbf{b)}$', fontsize=20)
            f.tight_layout()
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_power_phi_single(target_dir):  # plot of the power as a function of the phase offset
    colorlst = ['C0', 'C1', 'C2', 'C3']
    Ecouple_array = array([2.0, 8.0, 16.0, 32.0])

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            output_file_name = (
            target_dir + "Power_ATP_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

            plt.figure()
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.axhline(0, color='black', linewidth=1)

            # zero-barrier theory lines
            input_file_name = (
                    target_dir + "190624_Twopisweep_complete_set/processed_data/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2
            ax.axhline(power_y[17], color='C0', linewidth=2, label=None)

            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (target_dir + "191217_morepoints/processed_data/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(0, 4))
                    phase_array = data_array[:, 0]
                    power_y_array = -data_array[:, 1]

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

            f.text(0.55, 0.02, r'$\phi\ (\rm rev)$', fontsize=20, ha='center')
            plt.ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', fontsize=20)
            plt.xticks([0, pi / 9, 2 * pi / 9, pi / 3, 4 * pi / 9, 5 * pi / 9, 2 * pi / 3],
                       ['$0$', '', '', '$1/6$', '', '', '$1/3$'])
            plt.yticks([-0.0004, -0.0002, 0, 0.0002, 0.0004])

            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            f.tight_layout()
            f.subplots_adjust(bottom=0.12)
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_nn_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            output_file_name = (
                    target_dir + "power_nn_efficiency_Ecouple_plot_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")
            # flux plot
            # axarr[0, 0].axhline(0, color='black', linewidth=1, label='_nolegend_')  # line at zero

            # General data
            i = 0  # only use phase=0 data
            for j, num_min in enumerate(min_array):
                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                            usecols=(0, 4))
                        power_y = array(data_array[i, 1])
                        power_y_array = append(power_y_array, power_y)
                    except OSError:
                        print('Missing file flux')

                # Infinite coupling data
                input_file_name = (target_dir + "190729_Varying_n/processed_data/" +
                "Power_ATP_Ecouple_inf_no_n1_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name)
                    power_inf = array(data_array[j, 1])
                    # power_y_array = append(power_y_array, -power_inf)
                except OSError:
                    print('Missing file infinite coupling power')

                axarr[0, 0].plot(Ecouple_array, -power_y_array, marker='o', markersize=6, linestyle='-',
                                 color=color_lst[j])
                axarr[0, 1].plot([300], power_inf, marker='o', markersize=6, linestyle='-',
                                 color=color_lst[j])

            axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0, 0].yaxis.offsetText.set_fontsize(14)
            axarr[0, 0].tick_params(axis='y', labelsize=14)
            axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (t_{\rm sim}^{-1}) $', fontsize=20)
            axarr[0, 0].spines['right'].set_visible(False)
            axarr[0, 0].spines['top'].set_visible(False)
            axarr[0, 0].set_ylim((0, None))
            axarr[0, 0].set_xlim((8, None))

            axarr[0, 1].spines['right'].set_visible(False)
            axarr[0, 1].spines['top'].set_visible(False)
            axarr[0, 1].spines['left'].set_visible(False)
            axarr[0, 1].set_xticks([300])
            axarr[0, 1].set_xticklabels(['$\infty$'])
            axarr[0, 1].tick_params(axis='y', color='white')

            d = .015  # how big to make the diagonal lines in axes coordinates
            kwargs = dict(transform=axarr[0, 0].transAxes, color='k', clip_on=False)
            axarr[0, 0].plot((1 - 0.3 * d, 1 + 0.3 * d), (-d, +d), **kwargs)
            kwargs.update(transform=axarr[0, 1].transAxes)  # switch to the bottom axes
            axarr[0, 1].plot((-2.5 * d - 0.05, +2.5 * d - 0.05), (-d, +d), **kwargs)

            #########################################################
            # efficiency plot
            # axarr[1, 0].axhline(0, color='black', linewidth=1, label='_nolegend_')
            axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')

            for j, num_min in enumerate(min_array):
                eff_array = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "190729_Varying_n/processed_data/flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=(5))
                        eff_array = append(eff_array, data_array[0])
                    except OSError:
                        print('Missing file efficiency')

                # infinite coupling value
                # eff_array = append(eff_array, 0.5)
                axarr[1, 0].plot(Ecouple_array, eff_array/0.5, marker='o', markersize=6, linestyle='-',
                                 color=color_lst[j])
                axarr[1, 1].plot([300], 1, marker='o', markersize=6, linestyle='-',
                                 color=color_lst[j])

            axarr[1, 0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
            axarr[1, 0].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
            axarr[1, 0].set_xscale('log')
            axarr[1, 0].set_xlim((8, None))
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
    phase_array_1 = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319])
    phase_array_2 = array([0.0, 0.698132, 1.39626, 2.0944, 2.79253, 3.49066, 4.18879, 4.88692, 5.58505, 6.28319])
    phase_array_3 = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319])
    phase_array_6 = array([0.0, 2.0944, 4.18879, 6.28319])
    phase_array_12 = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 6.28319])
    Ecouple_array = array([16.0])

    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 5.5))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            output_file_name = (
                    target_dir + "power_efficiency_phi_vary_n_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")
            # flux plot
            axarr[0].axhline(0, color='black', linewidth=1)  # line at zero

            # zero-barrier theory lines
            input_file_name = (
                        target_dir + "190624_Twopisweep_complete_set/processed_data/"
                        + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2
            axarr[0].axhline(power_y[17], color='C0', linewidth=2, label='$0$')

            n = 1.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=4)
                    power_y = array(data_array)
                except OSError:
                    print('Missing file flux')

            power_y = append(power_y, power_y[0])
            axarr[0].plot(phase_array_1, -power_y, '-o', markersize=8, color=color_lst[0], label='$1$')

            n = 2.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=4)
                    power_y = array(data_array)
                except OSError:
                    print('Missing file flux')
            power_y = append(power_y, power_y[0])
            axarr[0].plot(phase_array_2, -power_y, '-o', markersize=8, color=color_lst[1], label='$2$')

            n = 3.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=4)
                    power_y = array(data_array)
                except OSError:
                    print('Missing file flux')
            axarr[0].plot(phase_array_3, -power_y[:7], '-o', markersize=8, color=color_lst[2], label='$3$')

            n = 6.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=4)
                    power_y = array(data_array)
                except OSError:
                    print('Missing file flux')
            axarr[0].plot(phase_array_6, -power_y[:4], '-o', markersize=8, color=color_lst[3], label='$6$')

            n = 12.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=4)
                    power_y = array(data_array)
                except OSError:
                    print('Missing file flux')
            power_y = delete(power_y, 5)
            axarr[0].plot(phase_array_12, -power_y, '-o', markersize=8, color=color_lst[4], label='$12$')

            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
            axarr[0].yaxis.offsetText.set_fontsize(14)
            axarr[0].tick_params(axis='both', labelsize=14)
            axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', fontsize=20)
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            axarr[0].set_ylim((0, None))
            axarr[0].set_xlim((0, 6.3))

            leg = axarr[0].legend(title=r'$n_{\rm o} = n_1$', fontsize=14, loc='lower center', frameon=False, ncol=3)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)

            #########################################################
            # efficiency plot
            axarr[1].axhline(0, color='black', linewidth=1)
            axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')
            axarr[1].set_aspect(1.5)

            n = 1.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                            target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple), usecols=5)
                    eff_array = data_array
                except OSError:
                    print('Missing file efficiency')
            eff_array = append(eff_array, eff_array[0])
            axarr[1].plot(phase_array_1, eff_array / (-psi_2 / psi_1), 'o', label='1', markersize=8, color=color_lst[0])

            n = 2.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple), usecols=5)
                    eff_array = data_array
                    # print(eff_array)
                except OSError:
                    print('Missing file efficiency')
            eff_array = append(eff_array, eff_array[0])
            axarr[1].plot(phase_array_2, eff_array / (-psi_2 / psi_1), 'o', label='2', markersize=8, color=color_lst[1])

            n = 3.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple), usecols=5)
                    eff_array = data_array
                    # print(eff_array)
                except OSError:
                    print('Missing file efficiency')
            eff_array = append(eff_array, eff_array[0])
            axarr[1].plot(phase_array_3, eff_array[:7] / (-psi_2 / psi_1), 'o', label='3', markersize=8,
                          color=color_lst[2])

            n = 6.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple), usecols=5)
                    eff_array = data_array
                    print(eff_array)
                except OSError:
                    print('Missing file efficiency')
            axarr[1].plot(phase_array_6, eff_array[:4] / (-psi_2 / psi_1), 'o', label='6', markersize=8,
                          color=color_lst[3])

            n = 12.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple), usecols=5)
                    eff_array = data_array
                    # print(eff_array)
                except OSError:
                    print('Missing file efficiency')
            eff_array = delete(eff_array, 5)
            axarr[1].plot(phase_array_12, eff_array / (-psi_2 / psi_1), 'o', label='12', markersize=8,
                           color=color_lst[4])

            axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
            axarr[1].set_ylim((0, 1.1))
            axarr[1].spines['right'].set_visible(False)
            axarr[1].spines['top'].set_visible(False)
            axarr[1].yaxis.offsetText.set_fontsize(14)
            axarr[1].tick_params(axis='both', labelsize=14)
            axarr[1].set_yticks([0, 0.5, 1.0])
            axarr[1].set_xticks([0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi])
            axarr[1].set_xticklabels(['$0$', '', '', '$1/2$', '', '', '$1$'])

            # leg = axarr[1].legend(title=r'$n_{\rm o} = n_1$', fontsize=14, loc='lower right', frameon=False)
            # leg_title = leg.get_title()
            # leg_title.set_fontsize(14)

            f.text(0.55, 0.02, r'$n \phi \ (\rm rev)$', fontsize=20, ha='center')
            f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
            f.text(0.03, 0.37, r'$\mathbf{b)}$', fontsize=20)
            f.tight_layout()
            # f.subplots_adjust(bottom=0.1)
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_n0_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "/power_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # flux plot
            # axarr[0, 0].axhline(0, color='black', linewidth=1, label='_nolegend_')  # line at zero

            # Ecouple_array_new = append(Ecouple_array, 300)
            # General data
            for j, num_min in enumerate(min_array):
                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, 3.0, Ecouple),
                            usecols=4)
                        power_y = array(data_array[j])
                        power_y_array = append(power_y_array, power_y)
                    except OSError:
                        print('Missing file flux')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, 3.0, Ecouple))

                axarr[0, 0].plot(Ecouple_array, -power_y_array, marker='o', markersize=6, linestyle='-',
                              color=color_lst[j])

                # infinite coupling result
                input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/" +
                                   "Power_ATP_Ecouple_inf_no_varies_n1_3.0_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat"
                                   )
                try:
                    data_array = loadtxt(input_file_name)
                    power_inf = array(data_array[j, 1])
                    # power_y_array = append(power_y_array, -power_inf)
                except OSError:
                    print('Missing file infinite coupling power')

                axarr[0, 1].plot([300], power_inf, marker='o', markersize=6, color=color_lst[j])

            axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0, 0].yaxis.offsetText.set_fontsize(14)
            axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (t_{\rm sim}^{-1}) $', fontsize=20)
            axarr[0, 0].spines['right'].set_visible(False)
            axarr[0, 0].spines['top'].set_visible(False)
            axarr[0, 0].set_ylim((0, None))
            axarr[0, 0].set_xlim((8, None))
            axarr[0, 0].tick_params(axis='both', labelsize=14)

            axarr[0, 1].spines['right'].set_visible(False)
            axarr[0, 1].spines['top'].set_visible(False)
            axarr[0, 1].spines['left'].set_visible(False)
            axarr[0, 1].set_xticks([300])
            axarr[0, 1].set_xticklabels(['$\infty$'])
            axarr[0, 1].tick_params(axis='y', color='white')

            d = .015  # how big to make the diagonal lines in axes coordinates
            kwargs = dict(transform=axarr[0, 0].transAxes, color='k', clip_on=False)
            axarr[0, 0].plot((1 - 0.3*d, 1 + 0.3*d), (-d, +d), **kwargs)
            kwargs.update(transform=axarr[0, 1].transAxes)  # switch to the bottom axes
            axarr[0, 1].plot((-2.5*d-0.05, +2.5*d-0.05), (-d, +d), **kwargs)

            #########################################################
            # efficiency plot
            # axarr[1, 0].axhline(0, color='black', linewidth=1, label='_nolegend_')
            axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')
            # axarr[1].set_aspect(0.5)

            for j, num_min in enumerate(min_array):
                eff_array = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, 3.0, Ecouple), usecols=5)
                        eff_array = append(eff_array, data_array[j])
                    except OSError:
                        print('Missing file efficiency')

                # eff_array = append(eff_array, 0.5)
                axarr[1, 0].plot(Ecouple_array, eff_array / (-psi_2 / psi_1), marker='o', markersize=6, linestyle='-',
                                 color=color_lst[j])
                axarr[1, 1].plot([300], [1], marker='o', markersize=6, color=color_lst[j])

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

            kwargs = dict(transform=axarr[1, 0].transAxes, color='k', clip_on=False)
            axarr[1, 0].plot((1 - 0.3*d, 1 + 0.3*d), (-2*d, +2*d), **kwargs)
            kwargs.update(transform=axarr[1, 1].transAxes)  # switch to the bottom axes
            axarr[1, 1].plot((-2.5*d-0.05, +2.5*d-0.05), (-2*d, +2*d), **kwargs)

            leg = axarr[1, 0].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o}$', fontsize=14, loc='best',
                                    frameon=False, ncol=3)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)

            f.text(0.77, 0.25, r'$n_1=3$', ha='center', fontsize=14)
            f.text(0.05, 0.92, r'$\mathbf{a)}$', ha='center', fontsize=20)
            f.text(0.05, 0.37, r'$\mathbf{b)}$', ha='center', fontsize=20)
            f.tight_layout()
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, 3.0))

if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/"
    # plot_power_efficiency_Ecouple(target_dir)
    # plot_power_Ecouple_grid(target_dir)
    # plot_power_efficiency_phi(target_dir)
    # plot_power_phi_single(target_dir)
    plot_nn_power_efficiency_Ecouple(target_dir)
    # plot_nn_power_efficiency_phi(target_dir)
    # plot_n0_power_efficiency_Ecouple(target_dir)