import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi, meshgrid, shape, amax, amin, zeros, round, append, exp, log, ones, sqrt, set_printoptions, isnan, delete
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc, ticker, colors, cm
from matplotlib.style import use
from scipy.integrate import trapz

N = 540
dx = 2*math.pi/N
positions = linspace(0, 2*math.pi-dx, N)
E0 = 2.0
E1 = 2.0
num_minima1 = 3.0
num_minima2 = 3.0

min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])[::-1]

psi1_array = array([2.0])
psi2_array = array([-1.33, -1.6, -1.78])6
# Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
# Ecouple_array_peak = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0])
Ecouple_array = array([11.31, 22.63, 45.25, 90.51])
# Ecouple_array_tot = array([2.0, 4.0, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 22.63, 24.0, 32.0, 45.25,
#                            64.0, 90.51, 128.0])
# Ecouple_array = array([16.0])
phase_array = array([0.0])
# phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533])
# phase_array_1 = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599])
# phase_array_2 = array([0.0, 0.5236, 1.0472, 1.5708, 2.0944, 2.6180])
# phase_array_3 = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533])
# phase_array_6 = array([0.0, 0.1745, 0.349066, 0.5236, 0.698132, 0.8727])
# phase_array_12 = array([0.0, 0.08727, 0.17453, 0.2618, 0.34633, 0.4363])
# phase_array_3 = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319])
# phase_array = phase_array_12
phi_ticks = [0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319]
phi_ticklabels = ['$0$', '', '', '$0.5$', '', '', '$1$']

colorlst = linspace(0, 1, len(min_array))
# Ecouple_labels = ['', '4', '', '16', '', '64', '', '\infty']
Ecouple_labels = ['2', '', '8', '', '32', '', '128']
phase_labels = ['$0$', '', '', '$\pi/12$', '', '', '$\pi/6$'][::-1] #n=12
n_labels = ['$1$', '$2$', '$3$', '$6$', '$12$'][::-1]
ylabels_eff = [0, 0.5, 1.0]
# phase_labels = ['0', '', '', '$\pi/3$'][::-1] #n=6
# phase_labels = ['0', '', '', '$\pi/3$', '', '', '$2 \pi/3$'][::-1] #n=3
# phase_labels = ['0', '', '', '$\pi/3$', '', '', '$2 \pi/3$', '', '', '$\pi$'][::-1] #n=2
# phase_labels = ['0', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$2 \pi$'][::-1] #n=1

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

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

def flux_power_efficiency(target_dir): #processing of raw data

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # phase_array = phase_array_1
            integrate_flux_X = empty(phase_array.size)
            integrate_flux_Y = empty(phase_array.size)
            integrate_power_X = empty(phase_array.size)
            integrate_power_Y = empty(phase_array.size)
            efficiency_ratio = empty(phase_array.size)

            for Ecouple in Ecouple_array:
                for ii, phase_shift in enumerate(phase_array):
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200220_moregrid" +
                                       "/reference_" +
                                       "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                    # if (num_minima1 == 3.0) and (Ecouple in Ecouple_array):  # this set of if's is for Ecouple and n varying data
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif (num_minima1 == 3.0) and (Ecouple in Ecouple_array_double):
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif (num_minima1 == 3.0) and (Ecouple in Ecouple_array_peak):
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190610_phaseoffset_extra" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif Ecouple in Ecouple_array:
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190924_no_vary_n1_3" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # else:
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200213_extrapoints" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")

                    # if num_minima1 == 3.0: # this set of if's is for the varying n and phase offset data
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif (num_minima1 == 12.0) and (phase_shift == 0.4363):
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200213_extrapoints" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif (num_minima1 == 2.0) and (phase_shift == 0.5236 or phase_shift == 1.5708 or phase_shift == 2.618):
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200213_extrapoints" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif (num_minima1 == 6.0) and (phase_shift == 0.1745 or phase_shift == 0.5236 or phase_shift == 0.8727):
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200213_extrapoints" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # else:
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190729_Varying_n/n1" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")

                    # if (num_minima1 == 3.0) and (Ecouple in Ecouple_array):  # this set of if's is for Ecouple and n varying data
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif (num_minima1 == 3.0) and (Ecouple in Ecouple_array_double):
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif (num_minima1 == 3.0) and (Ecouple in Ecouple_array_peak):
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190610_phaseoffset_extra" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # elif Ecouple in Ecouple_array:
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190729_Varying_n/n12" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    # else:
                    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200213_extrapoints" +
                    #                        "/reference_" +
                    #                        "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                    #                        "_outfile.dat")
                    output_file_name = (target_dir + "200220_moregrid/processed_data/" +
                                        "flux_power_efficiency_" +
                                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                                        "_outfile.dat")
                    
                    print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " +
                          f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")
                    
                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1,
                                                                    num_minima2, phase_shift),
                                             usecols=(0, 3, 4, 5, 6, 7, 8))
                        N = int(sqrt(len(data_array)))
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

                        flux_array = zeros((2, N, N))
                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array)/(dx*dx)

                        integrate_flux_X[ii] = (1./(2*pi))*trapz(trapz(flux_array[0, ...], dx=dx, axis=0), dx=dx)
                        integrate_flux_Y[ii] = (1./(2*pi))*trapz(trapz(flux_array[1, ...], dx=dx, axis=0), dx=dx)

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
    
                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), "w") as ofile:
                    for ii, phase_shift in enumerate(phase_array):
                        ofile.write(
                            f"{phase_shift:.15e}" + "\t"
                            + f"{integrate_flux_X[ii]:.15e}" + "\t"
                            + f"{integrate_flux_Y[ii]:.15e}" + "\t" 
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{efficiency_ratio[ii]:.15e}" + "\n")
                    ofile.flush()
            
def plot_efficiency_Ecouple_single(target_dir):#plot of the efficiency as a function of the coupling strength
    output_file_name = (target_dir + "efficiency_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phase_{4}" + "_.pdf")
    phaseoffset = 0.0
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                plt.figure()    
                ax=plt.subplot(111)
                ax.axhline(0, color='black', linewidth=1)
                ax.axhline(-psi_2/psi_1, color='grey', linewidth=1, linestyle='--')
                
                for i, n in enumerate(min_array):
                    eff_array = zeros(len(Ecouple_array))
                    for ii, Ecouple in enumerate(Ecouple_array):
                        input_file_name = (target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple), usecols=(5))
                            eff_array[ii] = data_array[0]#grab phase=0 result
                        except OSError:
                            print('Missing file')  
                            print(input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple))  
                        
                    plt.plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(colorlst[i]), label=n)
                plt.legend(title="$n_o=n_1$", loc='upper left')
                plt.xlabel('$E_{couple}$')
                plt.ylabel('$\eta$')
                plt.xscale('log')
                plt.ylim((None,1))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phaseoffset))     
                plt.close()       
        
def plot_flux_n_single(target_dir):#plot of the flux as a function of the number of minima
    output_file_name = (target_dir + "flux_n_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}" + "_.pdf")
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            
            for ii, Ecouple in enumerate(Ecouple_array):
                flux_x_array = []
                flux_y_array = []
                for n in min_array:
                    if n==3:
                        input_file_name = (target_dir + "190624_Twopisweep_complete_set/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    else:
                        input_file_name = (target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple), usecols=(0,1,2))
                        flux_x = data_array[0,1]
                        flux_x_array = append(flux_x_array, flux_x)
                        flux_y = data_array[0,2]
                        flux_y_array = append(flux_y_array, flux_y)
                
                    except OSError:
                        print('Missing file')
                    
                ax.plot(min_array, flux_y_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]), label=f'{Ecouple}')
                # ax.plot(min_array, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))

            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height*0.9])
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 0.8), shadow=False, title="$E_{couple}$")
            #plt.ylim((0,None))
            plt.xlabel('$n$')
            plt.ylabel('$P_{out}$')
            plt.xscale('log')
            #plt.xticks(ticklst, ticklabels)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2))    
            plt.close()

def plot_power_Ecouple_single(target_dir):#plot of power as a function of coupling strength
    output_file_name1 = (target_dir + "power_H_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phase_{4}" + "_.pdf")
    output_file_name2 = (target_dir + "power_ATP_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phase_{4}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array: #different plots for different forces
            if psi_1 > abs(psi_2):
                
                for j, phase in enumerate(phase_array): #different plots of different phase offsets
                    f1, ax1 = plt.subplots(1,1)
                    f2, ax2 = plt.subplots(1,1)
                    ax1.axhline(0, color='black', linewidth=1) #line at zero
                    ax2.axhline(0, color='black', linewidth=1)
                    # axarr.axhline(-0.00009, color='grey', linestyle='--', linewidth=1)#line to emphasize peak
                    
                    for i, n1 in enumerate(min_array): #different lines for different number of minima
                        power_x_array = zeros(len(Ecouple_array))
                        power_y_array = zeros(len(Ecouple_array))
                        for ii, Ecouple in enumerate(Ecouple_array): #different points in a line
                            input_file_name = (target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                            try:
                                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, n1, Ecouple), usecols=(3,4))
                                power_x_array[ii] = data_array[j,0]
                                power_y_array[ii] = data_array[j,1]
                            except OSError:
                                print('Missing file flux')
                                print(input_file_name.format(E0, E1, psi_1, psi_2,  num_minima2, Ecouple))
                        ax1.plot(Ecouple_array, power_x_array, 'o', color=plt.cm.cool(colorlst[i]), label=n1)
                        ax2.plot(Ecouple_array, power_y_array, 'o', color=plt.cm.cool(colorlst[i]), label=n1)

                    ax1.set_xscale('log')
                    ax1.set_xlabel('$E_{couple}$')
                    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    ax1.set_ylabel('$P_{ATP/ADP}$')
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['top'].set_visible(False)
                    f1.legend(loc='best', title='$n_o=n_1$')
                    f1.tight_layout()
                    
                    ax2.set_xscale('log')
                    ax2.set_xlabel('$E_{couple}$')
                    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    ax2.set_ylabel('$P_{ATP/ADP}$')
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    f2.legend(loc='best', title='$n_o=n_1$')
                    f2.tight_layout()

                    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, phase))
                    f2.savefig(output_file_name2.format(E0, E1, psi_1, psi_2, phase))
                    plt.close()

def plot_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "power_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 8))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # flux plot
            axarr[0].axhline(0, color='black', linewidth=0.5, label='_nolegend_')  # line at zero
            # maxpower = 0.000085247
            # axarr[0].axhline(maxpower, color='grey', linestyle=':', linewidth=1)  # line at infinite power coupling (calculated in Mathematica)
            # axarr[0].axhline(1, color='grey', linestyle=':', linewidth=1)#line at infinite power coupling
            # axarr[0].axvline(12, color='grey', linestyle=':', linewidth=1)  # lining up features in the two plots

            # General data
            i = 0  # only use phase=0 data
            for j, num_min in enumerate(min_array):
                power_x_array = []
                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                            usecols=(0, 3, 4))
                        power_x = array(data_array[i, 1])
                        power_y = array(data_array[i, 2])
                        power_x_array = append(power_x_array, power_x)
                        power_y_array = append(power_y_array, power_y)
                    except OSError:
                        print('Missing file flux')
                axarr[0].plot(Ecouple_array, -power_y_array, marker='o', markersize=6, linestyle='-')

            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0].yaxis.offsetText.set_fontsize(14)
            # axarr[0].set_yticks(ylabels_flux)
            # axarr[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axarr[0].tick_params(axis='y', labelsize=14)
            axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (t_{\rm sim}^{-1}) $', fontsize=20)
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            axarr[0].spines['bottom'].set_visible(False)
            # axarr[0].set_xlim((1.7, 135))

            leg = axarr[0].legend(n_labels, title=r'$n_{\rm o} = n_1$', fontsize=14, loc='lower right', frameon=False)
            leg_title = leg.get_title()
            leg_title.set_fontsize(14)

            #########################################################
            # efficiency plot
            axarr[1].axhline(0, color='black', linewidth=0.5)
            # axarr[1].set_aspect(0.5)
            # axarr[1].axvline(12, color='grey', linestyle=':', linewidth=1)

            for j, num_min in enumerate(min_array):
                eff_array = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "190729_Varying_n/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=(5))
                        eff_array = append(eff_array, data_array[0])
                    except OSError:
                        print('Missing file efficiency')
                axarr[1].plot(Ecouple_array, eff_array / (-psi_2 / psi_1), marker='o', markersize=6, linestyle='-')

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
            # f.subplots_adjust(hspace=0.01)
            f.tight_layout()
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_power_efficiency_phi(target_dir):  # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "power_efficiency_phi_vary_n_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 5.5))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # flux plot
            axarr[0].axhline(0, color='black', linewidth=1)  # line at zero

            # zero-barrier theory lines
            input_file_name = (
                        target_dir + "190624_Twopisweep_complete_set/processed_data/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            flux_x_array = array(data_array[:, 1])
            flux_y_array = array(data_array[:, 2])
            # power_x = flux_x_array * psi_1
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
                        usecols=(0, 3, 4))
                    phase_array = array(data_array[:, 0])
                    # power_x = array(data_array[:, 1])
                    power_y = array(data_array[:, 2])
                except OSError:
                    print('Missing file flux')
            # print(-power_y)
            power_y = append(power_y, power_y[0])
            # axarr[0].plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
            axarr[0].plot(phase_array_1, -power_y, '-o', markersize=8, color='C1', label='$1$')

            n = 2.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=(0, 3, 4))
                    phase_array = array(data_array[:, 0])
                    # power_x = array(data_array[:, 1])
                    power_y = array(data_array[:, 2])
                except OSError:
                    print('Missing file flux')
            power_y = append(power_y, power_y[0])
            # axarr[0].plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
            axarr[0].plot(phase_array_2, -power_y, '-o', markersize=8, color='C2', label='$2$')

            n = 3.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=(0, 3, 4))
                    phase_array = array(data_array[:, 0])
                    # power_x = array(data_array[:, 1])
                    power_y = array(data_array[:, 2])
                except OSError:
                    print('Missing file flux')
            print(phase_array[:7].size)
            print(phase_array_3.size)
            # power_y = append(power_y, power_y[0])
            # axarr[0].plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
            axarr[0].plot(phase_array_3, -power_y[:7], '-o', markersize=8, color='C3', label='$3$')

            n = 6.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=(0, 3, 4))
                    phase_array = array(data_array[:, 0])
                    # power_x = array(data_array[:, 1])
                    power_y = array(data_array[:, 2])
                except OSError:
                    print('Missing file flux')
            # axarr[0].plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
            axarr[0].plot(phase_array_6, -power_y[:4], '-o', markersize=8, color='C4', label='$6$')

            n = 12.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = (
                        target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, n, n, Ecouple),
                        usecols=(0, 3, 4))
                    phase_array = array(data_array[:, 0])
                    # power_x = array(data_array[:, 1])
                    power_y = array(data_array[:, 2])
                except OSError:
                    print('Missing file flux')
            power_y = delete(power_y, 5)
            # axarr[0].plot(Ecouple_array, psi_1*flux_x_array, 'o', color=plt.cm.cool(0))
            axarr[0].plot(phase_array_12, -power_y, '-o', markersize=8, color='C6', label='$12$')

            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0].set_xticks(phi_ticks)
            axarr[0].yaxis.offsetText.set_fontsize(14)
            axarr[0].tick_params(axis='both', labelsize=14)
            axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (t_{\rm sim}^{-1})$', fontsize=20)
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            axarr[0].set_ylim((0, None))

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
                    # print(eff_array)
                except OSError:
                    print('Missing file efficiency')
            eff_array = append(eff_array, eff_array[0])
            axarr[1].plot(phase_array_1, eff_array / (-psi_2 / psi_1), 'o', label='1', markersize=8, color='C1')

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
            axarr[1].plot(phase_array_2, eff_array / (-psi_2 / psi_1), 'o', label='2', markersize=8, color='C2')

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
            axarr[1].plot(phase_array_3, eff_array[:7] / (-psi_2 / psi_1), 'o', label='3', markersize=8, color='C3')

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
            axarr[1].plot(phase_array_6, eff_array[:4] / (-psi_2 / psi_1), 'o', label='6', markersize=8, color='C4')

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
            axarr[1].plot(phase_array_12, eff_array / (-psi_2 / psi_1), 'o', label='12', markersize=8, color='C6')

            axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
            axarr[1].set_xlim((-0.2, 6.4))
            axarr[1].set_ylim((0, 1.1))
            axarr[1].spines['right'].set_visible(False)
            axarr[1].spines['top'].set_visible(False)
            axarr[1].yaxis.offsetText.set_fontsize(14)
            axarr[1].tick_params(axis='both', labelsize=14)
            axarr[1].set_yticks(ylabels_eff)
            axarr[1].set_xticks(phi_ticks)
            axarr[1].set_xticklabels(phi_ticklabels)

            # leg = axarr[1].legend(title=r'$n_{\rm o} = n_1$', fontsize=14, loc='lower right', frameon=False)
            # leg_title = leg.get_title()
            # leg_title.set_fontsize(14)

            f.text(0.55, 0.02, r'$n \phi \ (\rm rev)$', fontsize=20, ha='center')
            f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
            f.text(0.03, 0.37, r'$\mathbf{b)}$', fontsize=20)
            f.tight_layout()
            # f.subplots_adjust(bottom=0.1)
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))

def plot_heatmap_power_Ecouple_n_scan(target_dir):
        
    input_file_name2 = (
        target_dir
        + "190729_Varying_n/processed_data"
        + "/flux_"
        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_inf"
        + "_outfile.dat"
        )
    power = zeros((psi2_array.size, psi1_array.size, Ecouple_array.size + 1, min_array.size))  #power = zeros((#plots in a row, #plots in a column, # of figures, # of entries in the x-axis, # of entries in the y-axis))
    
    for ii, psi_2 in enumerate(psi2_array):
        for jj, psi_1 in enumerate(psi1_array):
            for ee, Ecouple in enumerate(Ecouple_array):
                for nn, num_min in enumerate(min_array):
                    input_file_name = (
                        target_dir
                        + "190729_Varying_n/processed_data"
                        + "/flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}"
                        + "_outfile.dat"
                        )
                
                    power_X, power_Y = loadtxt(
                        input_file_name.format(
                            E0, E1, psi_1, psi_2, num_min, num_min, Ecouple
                            ),
                        unpack=True, usecols=(3,4)
                    )
                    
                    power[ii, jj, ee, nn] = power_Y[0] #grab phaseoffset=0 results
                    power[0, 0, ee, 0] = 'NaN'
                    
            min_array_out, flux_inf = loadtxt(
                input_file_name2.format(
                    E0, E1, psi_1, psi_2, num_minima2
                    ),
                unpack=True, usecols=(0,1)
            )
            
            power[ii, jj, ee + 1, :] = psi_2*flux_inf[::-1]

    limit=power[~(isnan(power))].__abs__().max()
    print(limit)
    # prepare figure
    fig1, ax1 = plt.subplots(psi1_array.size, psi2_array.size, figsize=(19,10), sharex='col', sharey='all')

    for ii, psi_2 in enumerate(psi2_array):
        for jj, psi_1 in enumerate(psi1_array):
            im1 = ax1[ii, jj].imshow(
                power[ii, jj, :, :].T,
                vmin=-limit, vmax=limit,
                cmap=plt.cm.get_cmap("coolwarm")
                )
                
            ax1[ii, jj].set_yticks(list(range(min_array.size)))
            ax1[ii, jj].set_yticklabels(min_labels)
            ax1[ii, jj].set_xticks(list(range(Ecouple_array.size+1)))
            ax1[ii, jj].set_xticklabels(Ecouple_labels)
            ax1[ii, jj].tick_params(labelsize=22)
            
            if (ii == 0):
                ax1[ii, jj].set_title(
                    "{}".format(psi_1), fontsize=20
                    )

            if (jj == psi2_array.size - 1):
                ax1[ii, jj].set_ylabel(
                    "{}".format(psi_2), fontsize=20
                    )
                ax1[ii, jj].yaxis.set_label_position("right")

    cbar_ticks = array([-5.0, -2.5, 0.0, 2.5, 5.0])*1e-4

    cax1 = fig1.add_axes([0.85, 0.25, 0.02, 0.5])
    cbar1 = fig1.colorbar(
        im1, cax=cax1, orientation='vertical', ax=ax1
    )
    cbar1.set_label(
        r'$\mathcal{P}_{\mathrm{ATP/ADP}}$', fontsize=32
        )
    cbar1.set_ticks(cbar_ticks)
    cbar1.formatter.set_powerlimits([0,0])
    cbar1.update_ticks()
    cbar1.ax.tick_params(labelsize=24)
    cbar1.ax.yaxis.offsetText.set_fontsize(24)
    cbar1.ax.yaxis.offsetText.set_x(5.0)

    #y-axis label
    fig1.text(
        0.8, 0.51,
        r'$\beta \psi_{1}$',
        fontsize=30, rotation='vertical', va='center', ha='center'
    )
    fig1.text(
        0.42, 0.03,
        r'$E_{\mathrm{couple}}$',
        fontsize=30, va='center', ha='center'
    )
    #x-axis label
    fig1.text(
            0.42, 0.93,
            r'$\beta \psi_{\mathrm{o}}$',
            fontsize=30, va='center', ha='center'
        )
    fig1.text(
            0.05, 0.48,
            r'$n_o$',
            fontsize=30, rotation='vertical', va='center', ha='center'
        )

    left = 0.1  # the left side of the subplots of the figure
    right = 0.75    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.88     # the top of the subplots of the figure
    fig1.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig1.savefig(
        "power_ATP_Ecouple_n_scan_small_E0_{0}_E1_{1}_n1_{2}_phase_{3}".format(
                E0, E1, num_minima1, 0.0
            )
        + "_figure.pdf",
        bbox_inches='tight'
        )
        
def plot_heatmap_power_Ecouple_phase_scan(target_dir):
        
    power = zeros((psi2_array.size, psi1_array.size, Ecouple_array.size, phase_array.size))  #power = zeros((#plots in a row, #plots in a column, # of figures, # of entries in the x-axis, # of entries in the y-axis))
    
    for ii, psi_2 in enumerate(psi2_array):
        for jj, psi_1 in enumerate(psi1_array):
            for ee, Ecouple in enumerate(Ecouple_array):
                for pp, phase in enumerate(phase_array):
                    input_file_name = (
                        target_dir
                        + "190729_Varying_n/processed_data"
                        + "/flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}"
                        + "_outfile.dat"
                        )
                
                    power_X, power_Y = loadtxt(
                        input_file_name.format(
                            E0, E1, psi_1, psi_2, num_minima1, num_minima1, Ecouple
                            ),
                        unpack=True, usecols=(3,4)
                    )
                    
                    power[ii, jj, ee, :(phase_array.size-2)] = power_Y[:(phase_array.size-2)] #grab phaseoffset=0 results
                    power[ii, jj, ee, -1] = power_Y[-1]
                    power[0, 0, ee, 0:2] = float('nan')
                

    limit=power[~(isnan(power))].__abs__().max()
    print(limit)
    # prepare figure
    fig1, ax1 = plt.subplots(psi1_array.size, psi2_array.size, figsize=(13,11), sharex='col', sharey='all')

    for ii, psi_2 in enumerate(psi2_array):
        for jj, psi_1 in enumerate(psi1_array):
            im1 = ax1[ii, jj].imshow(
                power[ii, jj, :, ::-1].T,
                vmin=-limit, vmax=limit,
                cmap=plt.cm.get_cmap("coolwarm")
                )
                
            ax1[ii, jj].set_yticks(list(range(phase_array.size)))
            ax1[ii, jj].set_yticklabels(phase_labels)
            ax1[ii, jj].set_xticks(list(range(Ecouple_array.size)))
            ax1[ii, jj].set_xticklabels(Ecouple_labels)
            ax1[ii, jj].tick_params(labelsize=22)
            
            if (ii == 0):
                ax1[ii, jj].set_title(
                    "{}".format(psi_1), fontsize=20
                    )

            if (jj == psi2_array.size - 1):
                ax1[ii, jj].set_ylabel(
                    "{}".format(psi_2), fontsize=20
                    )
                ax1[ii, jj].yaxis.set_label_position("right")

    cbar_ticks = array([-5.0, -2.5, 0.0, 2.5, 5.0])*1e-4

    cax1 = fig1.add_axes([0.85, 0.25, 0.02, 0.5])
    cbar1 = fig1.colorbar(
        im1, cax=cax1, orientation='vertical', ax=ax1
    )
    cbar1.set_label(
        r'$\mathcal{P}_{\mathrm{ATP/ADP}}$', fontsize=32
        )
    cbar1.set_ticks(cbar_ticks)
    cbar1.formatter.set_powerlimits([0,0])
    cbar1.update_ticks()
    cbar1.ax.tick_params(labelsize=24)
    cbar1.ax.yaxis.offsetText.set_fontsize(24)
    cbar1.ax.yaxis.offsetText.set_x(5.0)

    #y-axis label
    fig1.text(
        0.8, 0.51,
        r'$\beta \psi_{1}$',
        fontsize=30, rotation='vertical', va='center', ha='center'
    )
    fig1.text(
        0.42, 0.03,
        r'$E_{\mathrm{couple}}$',
        fontsize=30, va='center', ha='center'
    )
    #x-axis label
    fig1.text(
            0.42, 0.93,
            r'$\beta \psi_{\mathrm{o}}$',
            fontsize=30, va='center', ha='center'
        )
    fig1.text(
            0.05, 0.48,
            r'$\phi$',
            fontsize=30, rotation='vertical', va='center', ha='center'
        )

    left = 0.1  # the left side of the subplots of the figure
    right = 0.75    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.88     # the top of the subplots of the figure
    fig1.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig1.savefig(
        "power_ATP_Ecouple_phase_scan_small_E0_{0}_E1_{1}_n1_{2}".format(
                E0, E1, num_minima1
            )
        + "_figure.pdf",
        bbox_inches='tight'
        )
        
if __name__ == "__main__":
    target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/"
    flux_power_efficiency(target_dir)
    # plot_flux_n_single(target_dir)
    # plot_efficiency_Ecouple_single(target_dir)
    # plot_power_Ecouple_single(target_dir)
    # plot_power_efficiency_Ecouple(target_dir)
    # plot_power_efficiency_phi(target_dir)
    # plot_heatmap_power_Ecouple_n_scan(target_dir)
    # plot_heatmap_power_Ecouple_phase_scan(target_dir)
    