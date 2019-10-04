import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi, meshgrid, shape, amax, amin, zeros, round, append, exp, log
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

# psi1_array = array([0.0, 1.0, 2.0, 4.0])
# psi2_array = array([-4.0, -2.0, -1.0, 0.0])
psi1_array = array([4.0])
psi2_array = array([-2.0])

# Ecouple_array = array([2.0, 8.0, 32.0, 128.0]) #for grid plots
# Ecouple_array = array([256.0])
Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]) #twopisweep
Ecouple_array_extra = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0]) #extra measurements

#phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346, 2.79253, 3.14159, 3.49066, 3.83972, 4.18879, 4.53786, 4.88692, 5.23599, 5.58505, 5.93412, 6.28319]) #twopisweep
#phase_array = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599]) #n=1
#phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346, 2.79253]) #array for n=2
#phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533]) #array for n=6
#phase_array = array([0.0, 0.08727, 0.17453, 0.2618, 0.34633, 0.43633, 0.5236]) #array for n=12
#phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944]) #selection of twopisweep
#barrier_array = array([0.0, 2.0])
phase_array = array([0.0])

colorlist=linspace(0,1,len(min_array))
#label_lst=['0', '$\pi/9$', '$2\pi/9$', '$\pi/3$', '$4 \pi/9$', '$5 \pi/9$']
# size_lst=[12,10,8,6,4,2]
size_lst=[8,7,6,5,4,3]

#ticklabels=['0', '$\pi/6$', '$\pi/3$', '$\pi/2$', '$2 \pi/3$']
ticklabels=['0', '', '$2\pi/3$', '', '$4\pi/3$', '', '$2 \pi$'] #array for n=1
ticklst=linspace(0, 2*math.pi, 7)
ylabels_flux = [0, 0.0003, 0.0006]
ylabels_eff = [0, 0.5, 1]

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

def flux_power_efficiency():
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
                #for ii, phase_shift in enumerate(phase_array):
                    if num_minima==3.0:
                        N=360
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    else:
                        N=540
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190924_no_vary_n1_3" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
                    
                    output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/190924_no_vary_n1_3/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                    
                    print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " + f"Ecouple = {Ecouple}, num_minima1 = {num_minima}, num_minima2 = {num_minima2}")
                    flux_array = zeros((2,N,N))
                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima, num_minima2, phase_shift), usecols=(0,3,4,5,6,7,8))
                        prob_ss_array = data_array[:,0].reshape((N,N))
                        drift_at_pos = data_array[:,1:3].T.reshape((2,N,N))
                        diffusion_at_pos = data_array[:,3:].T.reshape((4,N,N))

                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)

                        flux_array = asarray(flux_array)/(dx*dx)

                        integrate_flux_X[ii] = (1./(2*pi))*trapz(trapz(flux_array[0,...], dx=dx, axis=1), dx=dx)
                        integrate_flux_Y[ii] = (1./(2*pi))*trapz(trapz(flux_array[1,...], dx=dx, axis=0), dx=dx)

                        #print(sum(integrate_flux_Y))
                        integrate_power_X[ii] = integrate_flux_X[ii]*psi_1
                        integrate_power_Y[ii] = integrate_flux_Y[ii]*psi_2
                    except:
                        print('Missing file')    
                if (abs(psi_1) <= abs(psi_2)):
                    efficiency_ratio = -(integrate_power_X/integrate_power_Y)
                else:
                    efficiency_ratio = -(integrate_power_Y/integrate_power_X)
    
                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), "w") as ofile:
                    for ii, num_minima in enumerate(min_array):
                    #for ii, phase_shift in enumerate(phase_array):
                        ofile.write(
                            f"{num_minima:.15e}" + "\t"
                            #"{phase_shift:.15e}" + "\t"
                            + f"{integrate_flux_X[ii]:.15e}" + "\t"
                            + f"{integrate_flux_Y[ii]:.15e}" + "\t" 
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{efficiency_ratio[ii]:.15e}" + "\n")
                    ofile.flush()
            
def plot_power_grid():
    output_file_name = ("power_Y_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,3,4))
                    #print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    power_x_array = data_array[:,1]
                    power_y_array = data_array[:,2]
    
                    axarr[i,j].plot(phase_array, power_y_array, color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')    
    #plt.legend(Ecouple_array, title="Ecouple")
    f.text(0.5, 0.04, '$\phi$', ha='center')
    f.text(0.04, 0.5, 'Output power', va='center', rotation='vertical')
    plt.xticks(ticklst, ticklabels)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))

def plot_power_single():
    output_file_name = ("Twopisweep/master_output_dir/power_XY_Ecouple_0.0_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()    
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("Twopisweep/master_output_dir/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,3,4))
                    #print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    power_x_array = data_array[:,1]
                    power_y_array = data_array[:,2]

                    plt.plot(phase_array, power_x_array, 'o', color=plt.cm.cool(colorlist[ii]), label=f'{Ecouple}')
                    plt.plot(phase_array, power_y_array, 'v', color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')      
            plt.legend(title="$E_{couple}$", loc='upper left')
            plt.xlabel('$\phi$')
            plt.ylabel('Power')
            plt.xticks(ticklst, ticklabels)
            #plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
    
def plot_efficiency_single():
    output_file_name = ("Twopisweep/master_output_dir/efficiency_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()    
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("Twopisweep/master_output_dir/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,5))
                    #print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    eff_array = data_array[:,1]

                    plt.plot(phase_array, eff_array, 'o', color=plt.cm.cool(colorlist[ii]), label=f'{Ecouple}')
                except OSError:
                    print('Missing file')    
            plt.legend(title="$E_{couple}$", loc='upper left')    
            plt.xlabel('$\phi$')
            plt.ylabel('$\eta$')
            #plt.grid()
            plt.xticks(ticklst, ticklabels)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            
def plot_efficiency_Ecouple_single():
    output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "/efficiency_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()    
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=1)
            # ax.axhline(0.5, color='grey', linestyle='--', linewidth=1)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
            try:
                print(input_file_name.format(psi_1, psi_2))
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                    
                Ecouple_array2 = array(data_array[1:,0])
                Ecouple_array2 = append(Ecouple_array2, 128.0) #add point to get a longer curve
                flux_x_array = array(data_array[1:,1])#1: is to skip the point at zero, which is problematic on a log scale
                flux_y_array = array(data_array[1:,2])
                flux_x_array = append(flux_x_array, flux_x_array[-1])#copy last point to add one
                flux_y_array = append(flux_y_array, flux_y_array[-1])
                
                if abs(psi_1) > abs(psi_2):
                    ax.plot(Ecouple_array2, -psi_2*flux_y_array/(psi_1*flux_x_array), '-', color=plt.cm.cool(0.99), linewidth=1.0)
                elif abs(psi_2) > abs(psi_1):
                    ax.plot(Ecouple_array2, -psi_1*flux_x_array/(psi_2*flux_y_array), '-', color=plt.cm.cool(0.99), linewidth=1.0)
            except:
                print('Missing data')
            
            eff_array = []
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190624_Twopisweep_complete_set/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(5))
                    eff_array = append(eff_array, data_array[0])
                except OSError:
                    print('Missing file')    
            plt.plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(0))
            # plt.legend(title="$E_{couple}$", loc='upper left') 
            plt.xlabel('$E_{couple}$')
            plt.ylabel('$\eta$')
            plt.xscale('log')
            plt.ylim((None,1))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #plt.grid()
            # plt.xticks(ticklst, ticklabels)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
    
def plot_efficiency_grid():
    output_file_name = ("efficiency_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            axarr[i,j].axhline(0, color='black', linewidth=1)#plot line at zero
            if abs(psi_1) > abs(psi_2):
                axarr[i,j].axhline(-psi_2/psi_1, linestyle='--', color='grey', linewidth=1)#plot line at zero
            elif abs(psi_2) > abs(psi_1):
                axarr[i,j].axhline(-psi_1/psi_2, linestyle='--', color='grey', linewidth=1)#plot line at zero
            
            for ii, Ecouple in enumerate(Ecouple_array):
                ##plot zero barrier theory
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
                try:
                    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                    Ecouple_array2 = array(data_array[1:,0])
                    Ecouple_array2 = append(Ecouple_array2, 128.0) #add point to get a longer curve
                    flux_x_array = array(data_array[1:,1])
                    flux_y_array = array(data_array[1:,2])
                    flux_x_array = append(flux_x_array, flux_x_array[-1])
                    flux_y_array = append(flux_y_array, flux_y_array[-1])
                    if abs(psi_1) > abs(psi_2):
                        axarr[i,j].plot(Ecouple_array2, -psi_2*flux_y_array/(psi_1*flux_x_array), '-', color=plt.cm.cool(0.99), linewidth=1.0)
                    elif abs(psi_2) > abs(psi_1):
                        axarr[i,j].plot(Ecouple_array2, -psi_1*flux_x_array/(psi_2*flux_y_array), '-', color=plt.cm.cool(0.99), linewidth=1.0)
                except:
                    print('Missing data')
                
            ##plot simulations zero barrier
            E0=0.0
            E1=0.0
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            eff_array=[]
            for ii, Ecouple in enumerate(Ecouple_array):
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,5))
                    print('Ecouple=%f'%Ecouple)
                    phase = data_array[0]
                    eff = data_array[1]
                    if i==j:
                        eff = -1
                    eff_array.append(eff)
                except OSError:
                    print('Missing file')
                    
            ##plot simulations 
            E0 = 2.0
            E1 = 2.0
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/190624_Twopisweep_complete_set/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            eff_array1=[]
            for ii, Ecouple in enumerate(Ecouple_array):
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,5))
                    print('Ecouple=%f'%Ecouple)
                    phase = data_array[0,0]
                    eff = data_array[0,1]
                    if i==j:
                        eff = -1
                    eff_array1.append(eff)
                except OSError:
                    print('Missing file')
            axarr[i,j].plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(0.99), markersize=4)  
            axarr[i,j].plot(Ecouple_array, eff_array1, 'o', color=plt.cm.cool(0), markersize=4)  
            axarr[i,j].spines['right'].set_visible(False)
            axarr[i,j].spines['top'].set_visible(False)
    #plt.legend(Ecouple_array, title="Ecouple")  
    plt.ylim(-0.33,1.0) 
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
    #plt.xticks(ticklst, ticklabels)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
    plt.close()
        
def plot_flux_grid():
    output_file_name = ("flux_y_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    #print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    flux_x_array = data_array[:,1]
                    flux_y_array = data_array[:,2]
    
                    axarr[i,j].plot(phase_array, flux_y_array, color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')    
            #plt.legend(Ecouple_array, title="Ecouple")    
    f.text(0.5, 0.04, '$\phi$', ha='center')
    f.text(0.04, 0.5, '$J_y$', va='center', rotation='vertical')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xticks(ticklst, ticklabels)
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
        
def plot_flux_single():
    output_file_name = ("flux_phi_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    phase_array = data_array[:,0]
                    flux_x_array = data_array[:,1]
                    flux_y_array = data_array[:,2]

                    ax.plot(phase_array, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]), label=f'{Ecouple}')
                    ax.plot(phase_array+0.5236, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+1.0472, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+1.5708, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+2.0944, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+2.618, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+3.1416, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+3.6652, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+4.1888, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+4.7124, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+5.236, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+5.7596, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    #ax.plot([6.28319], flux_x_array[0], 'o', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+0.5236, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+1.0472, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+1.5708, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+2.0944, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+2.618, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+3.1416, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+3.6652, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+4.1888, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+4.7124, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+5.236, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                    ax.plot(phase_array+5.7596, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')   
            
            # ##infinite coupling data
#             # input_file_name = ("Twopisweep/master_output_dir/processed_data/Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
#             data_array = loadtxt(input_file_name, usecols=(0,1))
#             #print('Ecouple=%f'%Ecouple)
#             phase_array = data_array[:,0]
#             flux_array = data_array[:,1]
#
#             ax.plot(phase_array, flux_array, '-', color=plt.cm.cool(colorlist[3]), label=f'$\infty$')

            plt.legend(title="$E_{couple}$", loc='upper left')    
            #f.text(0.5, 0.04, '$\phi$', ha='center')
            #f.text(0.04, 0.5, 'Output power', va='center', rotation='vertical')
            #plt.ylim((0,None))
            plt.xlabel('$\phi$')
            plt.ylabel('Flux')
            plt.xticks(ticklst, ticklabels)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))    
            plt.close()
            
def plot_flux_n_single():
    output_file_name = ("flux_n0_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}" + "_.pdf")
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            
            for ii, Ecouple in enumerate(Ecouple_array):
                flux_x_array = []
                flux_y_array = []
                # for n in min_array:
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(0,1,2))
                    flux_x_array = data_array[:,1]
                    # flux_x_array = append(flux_x_array, flux_x)
                    flux_y_array = data_array[:,2]
                    # flux_y_array = append(flux_y_array, flux_y)
                
                except OSError:
                    print('Missing file')
                    
                ax.plot(min_array, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]), label=f'{Ecouple}')
                ax.plot(min_array, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))

            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height*0.9])
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 0.8), shadow=False, title="$E_{couple}$")
            #plt.ylim((0,None))
            plt.xlabel('$n_o$')
            plt.ylabel('Flux')
            plt.xscale('log')
            #plt.xticks(ticklst, ticklabels)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2))    
            plt.close()            
            
def plot_flux_Ecouple_single():
    output_file_name = ("flux_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")
   
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            if abs(psi_1) >= abs(psi_2):
                plt.figure()
                ax=plt.subplot(111)
                ax.axhline(0, color='black', linewidth=1)#line at zero
                
                #for i, phase in enumerate(phase_array):
                i=0
                # #zero-barrier theory lines
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
            
                for j, num in enumerate(min_array):
                    num_minima1=num
                    num_minima2=3.0
                    flux_x_array=[]
                    flux_y_array=[]
                    for ii, Ecouple in enumerate(Ecouple_array):
                        # if num == 3.0:
                        input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190924_no_vary_n1_3/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}_Ecouple_{5}" + "_outfile.dat")
                        # else:
#                             input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190729_Varying_n/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima2, Ecouple), usecols=(0,1,2))
                            flux_x = data_array[j,1]
                            flux_y = data_array[j,2]
                            flux_x_array.append(flux_x)
                            flux_y_array.append(flux_y)
                        except OSError:
                            print('Missing file')
                    if num==3.0:
                        plt.axhline(y=flux_y_array[-1], ls='--', color='grey', linewidth=1)#dashed line to emphasize peak in flux
                    plt.plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(colorlist[j]), markersize=size_lst[j], label=num)
                    plt.plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(colorlist[j]), markersize=size_lst[j])

                # #add in a extra data points
                # flux_x_array=[]
                # flux_y_array=[]
                # E0=2.0
                # E1=2.0
                # for ii, Ecouple in enumerate(Ecouple_array_extra):
                #     input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190610_Extra_measurements/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
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
                #plt.grid(True, which='both')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phase_array[i]))
                plt.close()

def plot_flux_Ecouple_grid():
    output_file_name = ("flux_Ecouple_grid2_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all', sharey='all')
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            axarr[i,j].axhline(0, color='black', linewidth=1)#plot line at zero
            
            ##plot zero barrier theory
            try:
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                Ecouple_array2 = array(data_array[1:,0])
                Ecouple_array2 = append(Ecouple_array2, 128.0) #add point to get a longer curve
                flux_x_array = array(data_array[1:,1])
                flux_y_array = array(data_array[1:,2])
                flux_x_array = append(flux_x_array, flux_x_array[-1])
                flux_y_array = append(flux_y_array, flux_y_array[-1])
                axarr[i,j].plot(Ecouple_array2, flux_x_array, '--', color=plt.cm.cool(0.99), linewidth=1.0)
                axarr[i,j].plot(Ecouple_array2, flux_y_array, '-', color=plt.cm.cool(0.99), linewidth=1.0)
            except:
                print('Missing data')
            
            ##plot zero barrier data
            flux_x_array = []
            flux_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/processed_data/" + "flux_power_efficiency_" + "E0_0.0_E1_0.0_psi1_{0}_psi2_{1}_n1_{2}_n2_{3}_Ecouple_{4}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    #phase = data_array[0]
                    flux_x = data_array[1]
                    flux_x_array.append(flux_x)
                    flux_y = data_array[2]
                    flux_y_array.append(flux_y)
                except OSError:
                    print('Missing file')
            axarr[i,j].plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(0.99), markersize=3)
            axarr[i,j].plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(0.99), markersize=3)
            axarr[i,j].spines['right'].set_visible(False)
            axarr[i,j].spines['top'].set_visible(False) 
            
            ##plot barrier data
            # flux_x_array = []
            # flux_y_array = []
            # for ii, Ecouple in enumerate(Ecouple_array):
            #     input_file_name = ("processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            #     try:
            #         data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
            #         #phase = data_array[0,0] #only grabbing phi=0 data currently
            #         flux_x = data_array[0,1]
            #         flux_x_array.append(flux_x)
            #         flux_y = data_array[0,2]
            #         flux_y_array.append(flux_y)
            #     except OSError:
            #         print('Missing file')
            #
            # axarr[i,j].plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(0), markersize=4)
            # axarr[i,j].plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(0), markersize=4)
            # plt.xscale('log')
            # axarr[i,j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # axarr[i,j].spines['right'].set_visible(False)
            # axarr[i,j].spines['top'].set_visible(False)
            
                  
    
    f.text(0.52, 0.09, '$E_{couple}$', ha='center')
    f.text(0.09, 0.51, 'J', va='center', rotation='vertical')
    f.text(0.07, 0.75, '$\mu_{H^+}=1.0$', ha='center')
    f.text(0.06, 0.5, '2.0', ha='center')
    f.text(0.06, 0.27, '4.0', ha='center')
    f.text(0.3, 0.9, '$\mu_{ATP}=-1.0$', ha='center')
    f.text(0.53, 0.9, '$-2.0$', ha='center')
    f.text(0.77, 0.9, '$-4.0$', ha='center')
    plt.xscale('log')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.xticks(ticklst, ticklabels)
    f.tight_layout(pad=5.0, w_pad=1.0, h_pad=1.0)
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
    plt.close()
    
def plot_flux_contour():
    psi_1=0.0
    psi_2=0.0
    Ecouple=0.0
    phase=0.0
    flux_array = empty((2,N,N))
    input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    
    try:
        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,3,4,5,6,7,8))
        prob_ss_array = data_array[:,0].reshape((N,N))
        drift_at_pos = data_array[:,1:3].T.reshape((2,N,N))
        diffusion_at_pos = data_array[:,3:].T.reshape((4,N,N))

        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
        flux_array = asarray(flux_array)/(dx*dx)
        flux_x_array = flux_array[0]
        flux_y_array = flux_array[1]
        
        plt.contourf(positions, positions, flux_x_array)
        plt.colorbar()
    except OSError:
        print('Missing file')
    plt.show()   
    
def plot_flux():

    psi_1=0.0
    psi_2=0.0
    phase_shift=0.0
    Ecouple=0.0
    input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    output_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    integrate_flux_X = empty(N)
    integrate_flux_Y = empty(N)

    print("Calculating flux for " + f"psi_2 = {psi_2}, psi_1 = {psi_1}, " + f"Ecouple = {Ecouple}, phase = {phase_shift}")
    flux_array = empty((2,N,N))
    #print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift))
    data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift), usecols=(0,3,4,5,6,7,8))
    prob_ss_array = data_array[:,0].reshape((N,N))
    drift_at_pos = data_array[:,1:3].T.reshape((2,N,N))
    diffusion_at_pos = data_array[:,3:].T.reshape((4,N,N))

    calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)

    flux_array = asarray(flux_array)/(dx*dx)

    integrate_flux_X = trapz(flux_array[0,...], dx=dx, axis=1)
    integrate_flux_Y = trapz(flux_array[1,...], dx=dx, axis=0)
    
    plt.plot(positions, integrate_flux_X)
    #plt.ylim((0,None))
    plt.show()

def plot_flux_efficiency_Ecouple_single():
    output_file_name = ("flux_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")
    f,axarr=plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6,8))

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            #flux plot
            axarr[0].axhline(0, color='black', linewidth=1)#line at zero
            axarr[0].axvline(12, color='grey', linestyle='--', linewidth=1)
            
            i=0 #only use phase=0 data
            #zero-barrier theory lines
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:,0])
            Ecouple_array2 = append(Ecouple_array2, 128.0)
            flux_x_array = array(data_array[:,1])
            flux_y_array = array(data_array[:,2])
            flux_x_array = append(flux_x_array, flux_x_array[-1])
            flux_y_array = append(flux_y_array, flux_y_array[-1])
            axarr[0].plot(Ecouple_array2, flux_x_array, '--', color=plt.cm.cool(.99))
            axarr[0].plot(Ecouple_array2, flux_y_array, '-', color=plt.cm.cool(.99))

            #FP zero-barrier data points
            flux_x_array=[]
            flux_y_array=[]
            E0=0.0
            E1=0.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    flux_x = data_array[1]
                    flux_y = data_array[2]
                    flux_x_array.append(flux_x)
                    flux_y_array.append(flux_y)
                except OSError:
                    print('Missing file flux zero barrier')
            axarr[0].plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(.99))#, label=label_lst[i]
            axarr[0].plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(.99))
        
            flux_x_array=[]
            flux_y_array=[]
            E0=2.0
            E1=2.0
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190624_Twopisweep_complete_set/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    flux_x = data_array[i,1]
                    flux_y = data_array[i,2]
                    flux_x_array.append(flux_x)
                    flux_y_array.append(flux_y)
                except OSError:
                    print('Missing file flux')
            axarr[0].plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(0))
            axarr[0].plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(0))

            # add in a extra data points
            flux_x_array=[]
            flux_y_array=[]
            E0=2.0
            E1=2.0
            for ii, Ecouple in enumerate(Ecouple_array_extra):
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190610_Extra_measurements_Ecouple/processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    flux_x = data_array[i,1]
                    flux_y = data_array[i,2]
                    flux_x_array.append(flux_x)
                    flux_y_array.append(flux_y)
                except OSError:
                    print('Missing file flux extra points')
            axarr[0].plot(Ecouple_array_extra, flux_x_array, 'o', color=plt.cm.cool(colorlist[0]))
            axarr[0].plot(Ecouple_array_extra, flux_y_array, 'v', color=plt.cm.cool(colorlist[0]))
            axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            axarr[0].set_yticks(ylabels_flux)
            axarr[0].set_ylabel('J')
            axarr[0].spines['right'].set_visible(False)
            axarr[0].spines['top'].set_visible(False)
            
            
            #########################################################    
            #efficiency plot
            axarr[1].axhline(0, color='black', linewidth=1)
            axarr[1].axvline(12, color='grey', linestyle='--', linewidth=1)
            # ax.axhline(0.5, color='grey', linestyle='--', linewidth=1) 
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/Zero-energy_barriers/" + "Flux_Ecouple_Fx_{0}_Fy_{1}_theory.dat")
            try:
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                    
                Ecouple_array2 = array(data_array[1:,0])
                Ecouple_array2 = append(Ecouple_array2, 128.0) #add point to get a longer curve
                flux_x_array = array(data_array[1:,1])#1: is to skip the point at zero, which is problematic on a log scale
                flux_y_array = array(data_array[1:,2])
                flux_x_array = append(flux_x_array, flux_x_array[-1])#copy last point to add one
                flux_y_array = append(flux_y_array, flux_y_array[-1])
                
                if abs(psi_1) > abs(psi_2):
                    axarr[1].plot(Ecouple_array2, -psi_2*flux_y_array/(psi_1*flux_x_array), '-', color=plt.cm.cool(0.99), linewidth=1.0)
                elif abs(psi_2) > abs(psi_1):
                    axarr[1].plot(Ecouple_array2, -psi_1*flux_x_array/(psi_2*flux_y_array), '-', color=plt.cm.cool(0.99), linewidth=1.0)
            except:
                print('Missing data efficiency')
            
            eff_array = []
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190624_Twopisweep_complete_set/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(5))
                    eff_array = append(eff_array, data_array[0])
                except OSError:
                    print('Missing file efficiency')    
            axarr[1].plot(Ecouple_array, eff_array, 'o', color=plt.cm.cool(0))
            axarr[1].set_xlabel('$E_{couple}$')
            axarr[1].set_ylabel('$\eta$')
            axarr[1].set_xscale('log')
            axarr[1].set_ylim((None,1))
            axarr[1].spines['right'].set_visible(False)
            axarr[1].spines['top'].set_visible(False)
            axarr[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            axarr[1].set_yticks(ylabels_eff)
            plt.subplots_adjust(hspace=0)
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()
            
def plot_energy_flux():
    psi_1=4.0
    psi_2=-2.0
    Ecouple=128.0
    phase=0.0
    flux_array = empty((2,N,N))
    input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    output_file_name = ("Energy_flux_" + "Ecouple_{0}_E0_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_.pdf")
    
    fig, ax = plt.subplots()
    try:
        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,2,3,4,5,6,7,8))
        prob_ss_array = data_array[:,0].reshape((N,N))
        pot_array = data_array[:,1].reshape((N,N))
        drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
        diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
        
        plt.contourf(positions, positions, pot_array)
        plt.colorbar()

        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)

        flux_array = asarray(flux_array)/(dx*dx)
        flux_x_array = flux_array[0]
        flux_y_array = flux_array[1]
        
        M = 36
        fluxX = empty((M, M))
        fluxY = empty((M, M))
        for i in range(M):
            fluxX[i] = flux_x_array[int(N/M)*i, ::int(N/M)]
            fluxY[i] = flux_y_array[int(N/M)*i, ::int(N/M)]
           
        plt.quiver(positions[::int(N/M)], positions[::int(N/M)], fluxX, fluxY, units='xy')#headlength=1, headwidth=1, headaxislength=1
        
    except OSError:
        print('Missing file')
    
    plt.xlabel('X')
    plt.ylabel('Y')    
    plt.xticks(ticklst, ticklabels)
    plt.yticks(ticklst, ticklabels)
    target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/Twopisweep/master_output_dir"
    os.chdir(target_dir)
    fig.savefig(output_file_name.format(Ecouple, E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase)) 
    #plt.show()
    
def plot_energy_flux_grid():
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            #define arrays to add forces to energy landscapes
            Fx_array=empty((N,N))
            Fy_array=empty((N,N))
            Fx=psi_1*positions
            Fy=psi_2*positions
            for k in range(0,N):
                Fx_array[k]=Fx
                Fy_array[:,k]=Fy
    
            f,axarr=plt.subplots(3,7,sharex='all',sharey='all', figsize=(12,6))
            output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190530_Twopisweep/master_output_dir/Energy_flux_grid_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_big_.pdf")
    
            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            data_array = loadtxt(input_file_name.format(E0, 16.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=(0,2,3,4,5,6,7,8))
            prob_ss_array = data_array[:,0].reshape((N,N))
            pot_array = data_array[:,1].reshape((N,N))
            minpot=amin(pot_array)
            maxpot=amax(pot_array)
            drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
            diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
            flux_array = empty((2,N,N))
            calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
            flux_array = asarray(flux_array)/(dx*dx)
            flux_length_array = empty((N,N))
            flux_x_array = flux_array[0]
            flux_y_array = flux_array[1]
            flux_length_array = flux_x_array*flux_x_array + flux_y_array*flux_y_array
            maxflux=amax(flux_length_array)
            print(maxpot, math.sqrt(maxflux))
            for i, Ecouple in enumerate(Ecouple_array):
                for j, phase in enumerate(phase_array):
                    flux_array = empty((2,N,N))
                    input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            
                    try:
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,2,3,4,5,6,7,8))
                        prob_ss_array = data_array[:,0].reshape((N,N))
                        pot_array = data_array[:,1].reshape((N,N))
                        drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
                        diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
                        
                        if i==2 and j==0:
                            im = axarr[i,j].contourf(positions, positions, (pot_array.T), vmin=minpot, vmax=maxpot, cmap=plt.cm.cool)
                        else:
                            im2 = axarr[i,j].contourf(positions, positions, (pot_array.T), vmin=minpot, vmax=maxpot, cmap=plt.cm.cool)

                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array)/(dx*dx)
                        flux_x_array = flux_array[0]
                        flux_y_array = flux_array[1]
                
                        #select fewer arrows to draw
                        M = 18 #number of arrows in a row/ column, preferably a number such that N/M is an integer.
                        fluxX = empty((M, M))
                        fluxY = empty((M, M))
                        for k in range(M):
                            fluxX[k] = flux_x_array[int(N/M)*k, ::int(N/M)]
                            fluxY[k] = flux_y_array[int(N/M)*k, ::int(N/M)]

                        axarr[i,j].quiver(positions[::int(N/M)], positions[::int(N/M)], fluxX.T, fluxY.T, units='xy', angles='xy', scale_units='xy', scale=math.sqrt(maxflux))
                        axarr[i,j].set_aspect(aspect=1, adjustable='box-forced')

                    except OSError:
                        print('Missing file')

            f.text(0.45, 0.04, '$X$', ha='center')
            f.text(0.05, 0.5, '$Y$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.xticks(ticklst, ticklabels)
            plt.yticks(ticklst, ticklabels)
            f.subplots_adjust(right=0.8)
            cbar_ax=f.add_axes([0.85, 0.25, 0.03, 0.5])
            cbar=f.colorbar(im, cax=cbar_ax)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0,0))
            cbar.update_ticks()
            #f.subplots_adjust(wspace=0.01, hspace=0.1)

            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2)) 
            plt.close()
    
def plot_prob_flux():
    
    phase=0.0
    Ecouple=32.0
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "ProbSS_" + "Ecouple_{0}_E0_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}" + "_big_.pdf")
            
            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190812_ndata" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,2,3,4,5,6,7,8))
                prob_ss_array = data_array[:,0].reshape((N,N))
                maxprob=amax(prob_ss_array)
                drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
                diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
                flux_array = empty((2,N,N))
                calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                flux_array = asarray(flux_array)/(dx*dx)
                flux_length_array = empty((N,N))
                flux_x_array = flux_array[0]
                flux_y_array = flux_array[1]
                flux_length_array = flux_x_array*flux_x_array + flux_y_array*flux_y_array
                maxflux=amax(flux_length_array)
                print(maxprob, math.sqrt(maxflux))

                plt.contourf(positions, positions, prob_ss_array.T, vmin=0, vmax=maxprob, cmap=plt.cm.cool)

                #select fewer arrows to draw
                M = 18 #number of arrows in a row/ column, preferably a number such that N/M is an integer.
                fluxX = empty((M, M))
                fluxY = empty((M, M))
                for k in range(M):
                    fluxX[k] = flux_x_array[int(N/M)*k, ::int(N/M)]
                    fluxY[k] = flux_y_array[int(N/M)*k, ::int(N/M)]
                fluxzeros = zeros((M, M))    

                plt.quiver(positions[::int(N/M)], positions[::int(N/M)], fluxX.T, fluxY.T, units='xy', angles='xy', scale_units='xy', scale=math.sqrt(maxflux))  
                
            except OSError:
                print('Missing file')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.xticks(ticklst, ticklabels)
            plt.yticks(ticklst, ticklabels)
            ax.set_aspect(aspect=1.0)
            plt.savefig(output_file_name.format(Ecouple, E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()
                
def plot_prob_flux_grid():

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            f,axarr=plt.subplots(1,7,sharex='all',sharey='all', figsize=(12,2))
            output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "ProbEQ_unscaled_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
            
            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190729_varying_n/n1" + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            print(input_file_name.format(E0, 128.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
            data_array = loadtxt(input_file_name.format(E0, 128.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=(0,1,2,3,4,5,6,7,8))
            prob_ss_array = data_array[:,0].reshape((N,N))
            prob_eq_array = data_array[:,1].reshape((N,N))
            pot_array = data_array[:,2].reshape((N,N))
            maxprob=amax(pot_array)
            # maxpot=amax(pot_array)
            # drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
            # diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
            # flux_array = empty((2,N,N))
            # calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
            # flux_array = asarray(flux_array)/(dx*dx)
            # flux_length_array = empty((N,N))
            # flux_x_array = flux_array[0]
            # flux_y_array = flux_array[1]
            # flux_length_array = flux_x_array*flux_x_array + flux_y_array*flux_y_array
            # maxflux=amax(flux_length_array)
            # print(maxpot, math.sqrt(maxflux))
            
            #actually making subplots
            for i, Ecouple in enumerate(Ecouple_array):
                #for j, phase in enumerate(phase_array):
                phase=0.0
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                flux_array = empty((2,N,N))
                try:
                    data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,1,2,3,4,5,6,7,8))
                    prob_ss_array = data_array[:,0].reshape((N,N))
                    prob_eq_array = data_array[:,1].reshape((N,N))
                    pot_array = data_array[:,2].reshape((N,N))
                    drift_at_pos = data_array[:,3:5].T.reshape((2,N,N))
                    diffusion_at_pos = data_array[:,5:].T.reshape((4,N,N))
                    # if i==2 and j==0:
                    im = axarr[i].contourf(positions, positions, pot_array.T, vmin=0, cmap=plt.cm.cool)#plot energy landscape  
                    axarr[i].set_title(Ecouple)
                    #axarr[i].set_aspect('equal')
                    axarr[i].set_xlim([0,2*pi])
                    axarr[i].set_ylim([0,2*pi])
                    # else:
                    #     im2 = axarr[i,j].contourf(positions, positions, prob_ss_array.T, vmin=0, vmax=maxprob, cmap=plt.cm.cool)#plot energy landscape  
                    # axarr[i,j].plot(positions, positions, color='grey', linewidth=1.0)
                    # # calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                    # flux_array = asarray(flux_array)/(dx*dx)
                    # flux_x_array = (flux_array[0])
                    # flux_y_array = (flux_array[1])
                    #
                    # #select fewer arrows to draw
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
                     #axarr[i,j].grid()
                    
                except OSError:
                    print('Missing file')
            f.text(0.52, 0.94, '$E_{couple}$', ha='center')
            f.text(0.52, 0.01, '$F_o$', ha='center')
            f.text(0.04, 0.5, '$F_1$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.xticks(ticklst, ticklabels)
            plt.yticks(ticklst, ticklabels)

            f.tight_layout()
            top=0.8
            bottom=0.2
            right=0.9
            left=0.1
            f.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
            cbar_ax=f.add_axes([1.03*right, bottom, 0.02, top-bottom])
            cbar=f.colorbar(im, cax=cbar_ax)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0,0))
            cbar.update_ticks()
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()
            
def plot_condprob_grid():

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            f,axarr=plt.subplots(3,7,sharex='all',sharey='all', figsize=(12,6))
            output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "Condprob_10_flux_grid_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_big_low.pdf")
            
            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            data_array = loadtxt(input_file_name.format(E0, 16.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=(0))
            prob_ss_array = data_array.reshape((N,N))
            PXss = trapz(prob_ss_array, axis=1) #axis=0 gives P(f1), change to axis=1 for P(fo)
            cond_prob_array = empty((N,N))
            for i in range(0,N):
                for j in range(0,N):
                    cond_prob_array[i,j] = prob_ss_array[i,j]/PXss[i]#P(x2|x1)=P(x1,x2)/P(x1)
            maxprob=amax(cond_prob_array)
            print('maximum probability in grid:', maxprob)
            
            #actually making subplots
            for i, Ecouple in enumerate(Ecouple_array):
                for j, phase in enumerate(phase_array):
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0))
                        prob_ss_array = data_array.reshape((N,N))
                        PXss = trapz(prob_ss_array, axis=1)
                        cond_prob_array = empty((N,N))
                        for ii in range(0,N):
                            for jj in range(0,N):
                                cond_prob_array[ii,jj] = prob_ss_array[ii,jj]/PXss[ii]
                        if i==2 and j==0:
                            im = axarr[i,j].contourf(positions, positions, cond_prob_array.T, vmin=0, vmax=maxprob, cmap=plt.cm.cool) 
                        else:
                            im2 = axarr[i,j].contourf(positions, positions, cond_prob_array.T, vmin=0, vmax=maxprob, cmap=plt.cm.cool)
                        axarr[i,j].set_aspect(aspect=1, adjustable='box-forced')        
                    except OSError:
                        print('Missing file')
            f.text(0.5, 0.04, '$F_o$', ha='center')
            f.text(0.05, 0.5, '$F_1$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.xticks(ticklst, ticklabels)
            plt.yticks(ticklst, ticklabels)
            f.subplots_adjust(right=0.85)
            cbar_ax=f.add_axes([0.9, 0.25, 0.03, 0.5])
            cbar=f.colorbar(im, cax=cbar_ax)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0,0))
            cbar.update_ticks()
            f.subplots_adjust(wspace=0.2, hspace=0.05)
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()            
    
def plot_rel_flux():
    output_file_name = ("relflux_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
   
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)#line at zero
            
            for i, phase in enumerate(phase_array):                
                flux_x_array=[]
                flux_y_array=[]
                rel_flux=[]
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = ("processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        print("Plotting " + f"psi_2 = {psi_2}, psi_1 = {psi_1}, " + f"Ecouple = {Ecouple}, phase = {phase}")
                        data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                        flux_x = data_array[i,1]
                        flux_y = data_array[i,2]
                        flux_x_array.append(flux_x)
                        flux_y_array.append(flux_y)
                    except OSError:
                        print('Missing file')    
    
                try:
                    for k in range(0,len(Ecouple_array)):
                        rel_flux.append(flux_y_array[k]/flux_x_array[k])
                    plt.plot(Ecouple_array, rel_flux, 'o', color=plt.cm.cool(colorlist[i]), markersize=size_lst[i], label=label_lst[i])
                except:
                    print('Missing data')
            #add in second data set, extra measurements 
            # for i, phase in enumerate(phase_array):
            #     flux_x_array=[]
            #     flux_y_array=[]
            #     rel_flux=[]
            #     for ii, Ecouple in enumerate(Ecouple_array2):
            #         input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/190610_Extra_measurements/" + "processed_data/" + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            #         try:
            #             print("Plotting " + f"psi_2 = {psi_2}, psi_1 = {psi_1}, " + f"Ecouple = {Ecouple}, phase = {phase}")
            #             data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
            #             flux_x = data_array[i,1]
            #             flux_y = data_array[i,2]
            #             flux_x_array.append(flux_x)
            #             flux_y_array.append(flux_y)
            #         except OSError:
            #             print('Missing file')
            #
            #     for k in range(0,len(Ecouple_array2)):
            #         rel_flux.append(flux_y_array[k]/flux_x_array[k])
            #     plt.plot(Ecouple_array2, rel_flux, 'o', color=plt.cm.cool(colorlist[i]), markersize=size_lst[i])
                                
            plt.legend(title="$\phi$")  
            plt.xlabel('$E_{couple}$')
            plt.ylabel('Relative flux')
            plt.xscale('log')
            #plt.yscale('log')
            #plt.grid(True, which='both')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))     
            plt.close()
            
def plot_energy_prob_marg():
    
    output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/190624_Twopisweep_complete_set/" + "PMF_Pss_Y_norm_marg_grid_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset/" + "reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            force_X = positions*psi_1
            force_Y = positions*psi_2
            
            f,axarr=plt.subplots(3,7,sharex='all',sharey='all', figsize=(12,6))
            
            for i, Ecouple in enumerate(Ecouple_array):
                for j, phase in enumerate(phase_array):
                    try:
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,1,2))
                        prob_ss_array = data_array[:,0].reshape((N,N))
                        prob_eq_array = data_array[:,1].reshape((N,N))
                        pot_array = data_array[:,2].reshape((N,N))
        
                        prob_ss_X = trapz(prob_ss_array, dx=dx, axis=1)/ trapz(trapz(prob_ss_array, dx=dx, axis=1), dx=dx, axis=0)#axis=0 gives marg pdf of y, axis=1 gives marg pdf of x
                        prob_ss_Y = trapz(prob_ss_array, dx=dx, axis=0)/ trapz(trapz(prob_ss_array, dx=dx, axis=1), dx=dx, axis=0)#axis=0 gives marg pdf of y, axis=1 gives marg pdf of x
                        axarr[i,j].plot(positions, prob_ss_Y)
        
                        pot_X_w = trapz((pot_array-force_X[:,None]-force_Y[None,:])*prob_ss_array, dx=dx, axis=1)/prob_ss_X
                        pot_Y_w = trapz((pot_array-force_X[:,None]-force_Y[None,:])*prob_ss_array, dx=dx, axis=0)/prob_ss_Y
                        FreeE_X = trapz(exp(-(pot_array-force_X[:,None]-force_Y[None,:])), dx=dx, axis=1)
                        FreeE_Y = trapz(exp(-(pot_array-force_X[:,None]-force_Y[None,:])), dx=dx, axis=0)
                        PMF_X = trapz(exp(-(pot_array-force_X[:,None]-force_Y[None,:]))*prob_ss_array, dx=dx, axis=1)/trapz(prob_ss_array, dx=dx, axis=1)
                        PMF_Y = trapz(exp(-(pot_array-force_X[:,None]-force_Y[None,:]))*prob_ss_array, dx=dx, axis=0)/trapz(prob_ss_array, dx=dx, axis=0)
                        prob_eq_X = trapz(exp(-(pot_array)), dx=dx, axis=1)/ trapz(trapz(exp(-(pot_array)), dx=dx, axis=1), dx=dx, axis=0)#axis=0 gives marg pdf of y, axis=1 gives marg pdf of x
                        prob_eq_Y = trapz(exp(-(pot_array)), dx=dx, axis=0)/ trapz(trapz(exp(-(pot_array)), dx=dx, axis=1), dx=dx, axis=0)#axis=0 gives marg pdf of y, axis=1 gives marg pdf of x
                        
                        axarr[i,j].plot(positions, PMF_Y)
                        axarr[i,j].grid()
                        plt.xticks(ticklst, ticklabels)
                        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                        plt.yscale('log')
        
                    except OSError:
                        print('Missing file')
            
            f.text(0.05, 0.75, '$E_{couple}=0.0$', ha='center')
            f.text(0.05, 0.48, '$8.0$', ha='center')
            f.text(0.05, 0.22, '$16.0$', ha='center')
            f.text(0.18, 0.95, '$\phi=0.0$', ha='center')
            f.text(0.29, 0.95, '$\pi/9$', ha='center')
            f.text(0.40, 0.95, '$2\pi/9$', ha='center')
            f.text(0.51, 0.95, '$\pi/3$', ha='center')
            f.text(0.62, 0.95, '$4\pi/9$', ha='center')
            f.text(0.73, 0.95, '$5\pi/9$', ha='center')
            f.text(0.85, 0.95, '$2\pi/3$', ha='center')
            
            #plt.tight_layout()
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2)) 
            plt.close()
            
if __name__ == "__main__":
    # flux_power_efficiency()
    #plot_flux_single()
    # plot_flux_n_single()
    # plot_flux_Ecouple_single()
    #plot_flux_Ecouple_grid()
    #plot_flux_contour()
    #plot_flux_grid()
    #plot_power_single()
    #plot_power_grid()
    # plot_efficiency_single()
    # plot_efficiency_Ecouple_single()
    plot_flux_efficiency_Ecouple_single()
    #plot_efficiency_grid()
    #plot_energy_flux()
    #plot_energy_flux_grid()
    #plot_prob_flux()
    #plot_prob_flux_grid()
    #plot_condprob_grid()
    #plot_rel_flux()
    #plot_energy_prob_marg()