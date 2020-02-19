import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi, meshgrid, shape, amax, amin, zeros, round, append, exp, log, ones, sqrt
import math
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

N=360
dx=2*math.pi/N
positions=linspace(0,2*math.pi-dx,N)
E0=2.0
E1=2.0
num_minima1=3.0
num_minima2=3.0

min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])

psi1_array = array([4.0])
psi2_array = array([-2.0])
# psi1_array = array([1.0, 2.0, 4.0])
# psi2_array = array([-4.0, -2.0, -1.0])

Ecouple_array = array([16.0])
# Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
phase_array = array([0.0, 0.1745, 0.349066, 0.5236, 0.698132, 0.8727, 1.0472, 1.2217, 1.39626, 1.5708, 1.74533, 2.0944])
# phase_array = array([0.0])
n_labels = ['$1$', '$2$', '$3$', '$6$', '$12$']
ylabels_eff = [0, 0.5, 1.0]

colorlst = linspace(0, 1, len(Ecouple_array))

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

def flux_power_efficiency(target_dir):
    phase_shift=0.0
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # if abs(psi_1) >= abs(psi_2):
                   
            integrate_flux_X = empty(min_array.size)
            integrate_flux_Y = empty(min_array.size)
            integrate_power_X = empty(min_array.size)
            integrate_power_Y = empty(min_array.size)
            efficiency_ratio = empty(min_array.size)

            for Ecouple in Ecouple_array:
                for ii, num_minima in enumerate(min_array):
                    if num_minima==3.0:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    else:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190924_no_vary_n1_3" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    
                    output_file_name = target_dir + "190924_no_vary_n1_3/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat"
                    
                    print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " + f"Ecouple = {Ecouple}, num_minima1 = {num_minima}, num_minima2 = {num_minima2}")
                    
                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima, num_minima2, phase_shift), usecols=(0,3,4,5,6,7,8))
                        N = int(sqrt(len(data_array)))
                        print(N)
                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

                        flux_array = zeros((2, N, N))
                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array)/(dx*dx)

                        integrate_flux_X[ii] = (1./(2*pi))*trapz(trapz(flux_array[0, ...], dx=dx, axis=1), dx=dx)
                        integrate_flux_Y[ii] = (1./(2*pi))*trapz(trapz(flux_array[1, ...], dx=dx, axis=0), dx=dx)

                        # print(sum(integrate_flux_Y))
                        integrate_power_X[ii] = integrate_flux_X[ii]*psi_1
                        integrate_power_Y[ii] = integrate_flux_Y[ii]*psi_2
                    except:
                        print('Missing file')    
                if abs(psi_1) <= abs(psi_2):
                    efficiency_ratio = -(integrate_power_X/integrate_power_Y)
                else:
                    efficiency_ratio = -(integrate_power_Y/integrate_power_X)
    
                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), "w") as ofile:
                    for ii, num_minima in enumerate(min_array):
                        ofile.write(
                            f"{num_minima:.15e}" + "\t"
                            + f"{integrate_flux_X[ii]:.15e}" + "\t"
                            + f"{integrate_flux_Y[ii]:.15e}" + "\t" 
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{efficiency_ratio[ii]:.15e}" + "\n")
                    ofile.flush()
                    
def plot_power_Ecouple_single(target_dir): # plot of power as a function of coupling strength
    output_file_name = (target_dir + "/power_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_no_{4}" + "_log_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                f,axarr=plt.subplots(1, 1, sharex='all', sharey='none', figsize=(6,4))
                axarr.axhline(0, color='black', linewidth=1)#line at zero
                # axarr.axhline(-0.00009, color='grey', linestyle='--', linewidth=1)#line to emphasize peak

                for i, n1 in enumerate(min_array): #different lines for different number of minima
                    flux_x_array = zeros(len(Ecouple_array))
                    flux_y_array = zeros(len(Ecouple_array))
                    for ii, Ecouple in enumerate(Ecouple_array): #different points in a line
                        input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(1,2))
                            flux_x_array[ii] = data_array[i,0]
                            flux_y_array[ii] = data_array[i,1]
                        except OSError:
                            print('Missing file flux')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple))
                    power_x = flux_x_array*psi_1
                    power_y = flux_y_array*psi_2
                    axarr.plot(Ecouple_array, power_y, 'o', color=plt.cm.cool(colorlst[i]), label=n1)

                axarr.set_xscale('log')
                axarr.set_xlabel('$E_{couple}$')
                axarr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                axarr.set_ylabel('$P_{ATP/ADP}$')
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                plt.legend(loc='best', title='$n_o$')
                plt.tight_layout()

                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1))
                plt.close()

def plot_power_Ecouple_scaled(target_dir):#plot of scaled power as a function of coupling strength
    output_file_name = (target_dir + "/power_scaled_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_no_{4}" + "_log_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                f,axarr=plt.subplots(1, 1, sharex='all', sharey='none', figsize=(6,4))
                axarr.axhline(0, color='black', linewidth=1)#line at zero
                axarr.axhline(1, color='grey', linestyle='--', linewidth=1)#line to emphasize peak
                
                for i, n1 in enumerate(min_array): #different lines for different number of minima
                    flux_x_array = zeros(len(Ecouple_array))
                    flux_y_array = zeros(len(Ecouple_array))
                    
                    #infinite coupling
                    input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_inf" + "_outfile.dat")
                    try:
                        data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2), usecols=(1))
                        flux = data_array[i]
                    except:
                        print('Missing infinite flux data')
                    power_inf_y = flux*psi_2
                    
                    for ii, Ecouple in enumerate(Ecouple_array): #different points in a line
                        input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(1,2))
                            flux_x_array[ii] = data_array[i,0]
                            flux_y_array[ii] = data_array[i,1]
                        except OSError:
                            print('Missing file flux')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple))
                    power_x = flux_x_array*psi_1
                    power_y = flux_y_array*psi_2
                    axarr.plot(Ecouple_array, power_y/power_inf_y, 'o', color=plt.cm.cool(colorlst[i]), label=n1)

                axarr.set_xscale('log')
                axarr.set_xlabel('$E_{couple}$')
                axarr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                axarr.set_ylabel('$P_{ATP/ADP}/P_{ATP/ADP}^{\infty}$')
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                plt.legend(loc='best', title='$n_o$')
                plt.tight_layout()

                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1))
                plt.close()

def plot_power_n1_single(target_dir):#plot of power as a function of the number of minima of F1
    output_file_name = (target_dir + "/power_no_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_no_{4}" + "_log_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                f,axarr=plt.subplots(1, 1, sharex='all', sharey='none', figsize=(6,4))
                axarr.axhline(0, color='black', linewidth=1)#line at zero
                # axarr.axhline(1, color='grey', linestyle='--', linewidth=1)#line to emphasize peak
                input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                
                for ii, Ecouple in enumerate(Ecouple_array): #different points in a line
                    power_x_array = zeros(len(min_array))
                    power_y_array = zeros(len(min_array))
                    for i, n1 in enumerate(min_array): #different lines for different number of minima
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(3,4))
                            power_x_array = data_array[:,0]
                            power_y_array = data_array[:,1]
                        except OSError:
                            print('Missing file flux')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple))
                    axarr.plot(min_array, power_y_array, 'o', color=plt.cm.cool(colorlst[ii]), label=Ecouple)

                axarr.set_xscale('log')
                axarr.set_xlabel('$n_o$')
                axarr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                axarr.set_ylabel('$P_{ATP/ADP}$')
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                plt.legend(loc='best', title='$E_{couple}$')
                plt.tight_layout()

                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1))
                plt.close()
                
def plot_power_n1_scaled(target_dir):#plot of scaled power as a function of the number of minima of F1
    output_file_name = (target_dir + "/power_scaled_no_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_no_{4}" + "_log_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                f,axarr=plt.subplots(1, 1, sharex='all', sharey='none', figsize=(6,4))
                axarr.axhline(0, color='black', linewidth=1)#line at zero
                axarr.axhline(1, color='grey', linestyle='--', linewidth=1)#line to emphasize peak
                
                for ii, Ecouple in enumerate(Ecouple_array): #different points in a line
                    power_x_array = zeros(len(min_array))
                    power_y_array = zeros(len(min_array))
                    
                    for i, n1 in enumerate(min_array): #different lines for different number of minima
                        #infinite coupling
                        input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_inf" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2), usecols=(1))
                            flux_array = data_array
                        except:
                            print('Missing infinite flux data')
                        power_inf_y = flux_array*psi_2
                    
                        input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(3,4))
                            power_x_array = data_array[:,0]
                            power_y_array = data_array[:,1]
                        except OSError:
                            print('Missing file flux')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple))
                    axarr.plot(min_array, power_y_array/power_inf_y, 'o', color=plt.cm.cool(colorlst[ii]), label=Ecouple)

                axarr.set_xscale('log')
                axarr.set_xlabel('$n_o$')
                axarr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                axarr.set_ylabel('$P_{ATP/ADP}/P_{ATP/ADP}^{\infty}$')
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                plt.legend(loc='best', title='$E_{couple}$')
                plt.tight_layout()

                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1))
                plt.close()

def plot_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "/power_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 8))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # flux plot
            axarr[0].axhline(0, color='black', linewidth=1, label='_nolegend_')  # line at zero

            # General data
            for j, num_min in enumerate(min_array):
                power_x_array = []
                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (
                                target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, 3.0, Ecouple),
                            usecols=(0, 3, 4))
                        power_x = array(data_array[j, 1])
                        power_y = array(data_array[j, 2])
                        power_x_array = append(power_x_array, power_x)
                        power_y_array = append(power_y_array, power_y)
                    except OSError:
                        print('Missing file flux')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, 3.0, Ecouple))
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

            leg = axarr[0].legend(n_labels, title=r'$n_{\rm o}$', fontsize=14, loc='lower right', frameon=False, ncol=1)
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
                                target_dir + "/190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, 3.0, Ecouple), usecols=(5))
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
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, 3.0))

def plot_efficiency_Ecouple_single(target_dir):#plot of the efficiency as a function of the coupling strength
    output_file_name = (target_dir + "/efficiency_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_no_{4}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                plt.figure()
                ax=plt.subplot(111)
                ax.axhline(0, color='black', linewidth=1)
                ax.axhline(-psi_2/psi_1, color='grey', linestyle='--', linewidth=1)
                
                for i, n1 in enumerate(min_array):
                    eff_array = zeros(len(Ecouple_array))
                    for ii, Ecouple in enumerate(Ecouple_array):
                        input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(5))
                            eff_array[ii] = data_array[i]
                        except OSError:
                            print('Missing file')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple))
                    plt.plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(colorlst[i]), label=n1)
                plt.legend(title="$n_o$", loc='best')
                plt.xlabel('$E_{couple}$')
                plt.ylabel('$\eta$')
                plt.xscale('log')
                plt.ylim((None,1))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                plt.close()
                
def plot_efficiency_n1_single(target_dir):#plot of the efficiency as a function of the coupling strength
    output_file_name = (target_dir + "/efficiency_no_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_no_{4}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                plt.figure()
                ax=plt.subplot(111)
                ax.axhline(0, color='black', linewidth=1)
                ax.axhline(-psi_2/psi_1, color='grey', linestyle='--', linewidth=1)
                input_file_name = (target_dir + "/190924_no_vary_n1_3/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                
                for ii, Ecouple in enumerate(Ecouple_array): 
                    eff_array = zeros(len(min_array))
                    for i, n1 in enumerate(min_array):
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(5))
                            eff_array = data_array
                        except OSError:
                            print('Missing file')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple))
                    plt.plot(min_array, eff_array, 'o', color=plt.cm.cool(colorlst[ii]), label=Ecouple)
                plt.legend(title="$E_{couple}$", loc='best')
                plt.xlabel('$n_o$')
                plt.ylabel('$\eta$')
                plt.xscale('log')
                plt.ylim((None,1))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
                plt.close()
                    
if __name__ == "__main__":
    target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython"
    flux_power_efficiency(target_dir)
    # plot_power_Ecouple_single(target_dir)
    # plot_power_Ecouple_scaled(target_dir)
    # plot_efficiency_Ecouple_single(target_dir)
    # plot_efficiency_n1_single(target_dir)
    # plot_power_n1_single(target_dir)
    # plot_power_n1_scaled(target_dir)
    # plot_power_efficiency_Ecouple(target_dir)