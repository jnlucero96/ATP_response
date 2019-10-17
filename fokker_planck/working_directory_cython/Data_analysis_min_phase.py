import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi, meshgrid, shape, amax, amin, zeros, round, append, exp, log, ones, sqrt
import math
import matplotlib.pyplot as plt
from scipy.integrate import trapz

N=360
dx=2*math.pi/N
positions=linspace(0,2*math.pi-dx,N)
E0=2.0
E1=2.0
num_minima1=3.0
num_minima2=3.0

min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])

psi1_array = array([4.0])
# psi1_array = array([0.0, 1.0, 2.0, 4.0])
# psi2_array = array([-4.0, -2.0, -1.0, 0.0])
psi2_array = array([-2.0])
Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]) 
# phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346]) 
phase_array_1 = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599])
phase_array_2 = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346, 2.79253])
phase_array_3 = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346, 2.79253])
phase_array_6 = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533])
phase_array_12 = array([0.0, 0.08727, 0.17453, 0.2618, 0.34633, 0.5236])
min_array = phase_array_3

colorlst = linspace(0,1,len(Ecouple_array))

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
                for ii, phase_shift in enumerate(min_array):
                    if num_minima1==3.0:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    else:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190729_varying_n" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    
                    output_file_name = (target_dir + "/190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    
                    print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " + f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")
                    
                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift), usecols=(0,3,4,5,6,7,8))
                        N = int(sqrt(len(data_array)))
                        # print(N)
                        prob_ss_array = data_array[:,0].reshape((N,N))
                        drift_at_pos = data_array[:,1:3].T.reshape((2,N,N))
                        diffusion_at_pos = data_array[:,3:].T.reshape((4,N,N))

                        flux_array = zeros((2,N,N))
                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array)/(dx*dx)

                        integrate_flux_X[ii] = (1./(2*pi))*trapz(trapz(flux_array[0,...], dx=dx, axis=1), dx=dx)
                        integrate_flux_Y[ii] = (1./(2*pi))*trapz(trapz(flux_array[1,...], dx=dx, axis=0), dx=dx)

                        integrate_power_X[ii] = integrate_flux_X[ii]*psi_1
                        integrate_power_Y[ii] = integrate_flux_Y[ii]*psi_2
                    except:
                        print('Missing file')    
                        
                if (abs(psi_1) <= abs(psi_2)):
                    efficiency_ratio = -(integrate_power_X/integrate_power_Y)
                else:
                    efficiency_ratio = -(integrate_power_Y/integrate_power_X)
    
                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), "w") as ofile:
                    for ii, phase_shift in enumerate(min_array):
                        ofile.write(
                            f"{phase_shift:.15e}" + "\t"
                            + f"{integrate_flux_X[ii]:.15e}" + "\t"
                            + f"{integrate_flux_Y[ii]:.15e}" + "\t" 
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{efficiency_ratio[ii]:.15e}" + "\n")
                    ofile.flush()
            
def plot_efficiency_Ecouple_single(target_dir):#plot of the efficiency as a function of the coupling strength
    output_file_name = (target_dir + "efficiency_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if psi_1 > abs(psi_2):
                plt.figure()    
                ax=plt.subplot(111)
                ax.axhline(0, color='black', linewidth=1)
            
                #General results
                eff_array = zeros(len(Ecouple_array))
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = (target_dir + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(5))
                        eff_array[ii] = data_array[0]#grab phi=0 result
                    except OSError:
                        print('Missing file')  
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))  
                        
                plt.plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(0))
                # plt.legend(title="$E_{couple}$", loc='upper left') 
                plt.xlabel('$E_{couple}$')
                plt.ylabel('$\eta$')
                plt.xscale('log')
                plt.ylim((None,1))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))     
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
                    
if __name__ == "__main__":
    target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/"
    # flux_power_efficiency(target_dir)
    # plot_flux_n_single(target_dir)
    plot_efficiency_Ecouple_single(target_dir)