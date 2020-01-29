from numpy import loadtxt, sqrt, cos, pi, linspace, zeros, array
import matplotlib.pyplot as plt
import scipy.integrate as trapz

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

N = 360
dx = 2*pi/N
psi1_array = [4.0, 2.0, 1.0]
psi_2 = -1.0
Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
# Ecouple_array = array([0.0])
num_minima = 3.0
E0 = 2.0
E1 = 2.0
phase_shift = 0.0
xlst = linspace(0, 2*pi, N)

def coupling_energy(Ecouple, x, y):
    return -0.5*Ecouple*cos(x-y)

def energy_flow_li_ma(target_dir): #processing of raw data

    plt.figure()
    f1, ax1 = plt.subplots(1, 1, figsize=(5, 4))

    for psi_1 in psi1_array:
        energyflow = zeros(Ecouple_array.size)
        energyflow_xy = zeros((N, N))
        couple_energy = zeros((N, N))
        for k, Ecouple in enumerate(Ecouple_array):
            input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" +
                               "/reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                               "_outfile.dat")

            output_file_name = (target_dir + "energyflow_" +
                                "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" +
                                "_outfile.pdf")

            print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " +
                  f"Ecouple = {Ecouple}, num_minima1 = {num_minima}, num_minima2 = {num_minima}")

            try:
                # print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima, num_minima, phase_shift))
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima, num_minima, phase_shift),
                    usecols=0)
                prob_ss_array = data_array[:].reshape((N,N))
            except OSError:
                print('Missing file')

            for i, x in enumerate(xlst):
                for j, y in enumerate(xlst):
                    couple_energy[i, j] = coupling_energy(Ecouple, x, y)
                    energyflow_xy[i, j] = couple_energy[i, j]*prob_ss_array[i, j]

            energyflow[k] = -trapz.trapz(trapz.trapz(energyflow_xy, dx=dx), dx=dx)

        ax1.plot(Ecouple_array, energyflow, marker='.', linestyle='-', label=psi_1)
    ax1.set_xlabel("$E_{\\rm couple}$")
    ax1.set_ylabel("Energy flow")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.legend(title='$\mu_{\\rm o}$')
    # ax1.set_xscale('log')

    f1.tight_layout()
    f1.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima, num_minima))
    plt.close()

if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/" + \
                 "working_directory_cython/"
    energy_flow_li_ma(target_dir)