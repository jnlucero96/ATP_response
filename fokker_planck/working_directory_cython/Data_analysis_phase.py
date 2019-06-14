import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi, meshgrid, shape, amax, zeros, round
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

#psi1_array = array([4.0, 2.0])
#psi2_array = array([-1.0])
psi1_array = array([0.0, 1.0, 2.0, 4.0, 8.0])
psi2_array = array([-8.0, -4.0, -2.0, -1.0, 0.0])

Ecouple_array = array([0.0, 8.0, 16.0]) #for grid plots
#Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]) #twopisweep
#Ecouple_array = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0]) #extra measurements
#phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346, 2.79253, 3.14159, 3.49066, 3.83972, 4.18879, 4.53786, 4.88692, 5.23599, 5.58505, 5.93412, 6.28319]) #twopisweep

#phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533]) #twopisweep, only get data of first 1/3 
#phase_array = round(array([0.0, 0.628319, 1.25664, 1.88496, 2.51327, 3.14159])/3 ,2) #phaseoffset data, divide by 3 to get true angle
phase_array = array([0.0, 0.628319, 1.25664, 1.88496, 2.51327, 3.14159])
#phase_array = array([0.0, 0.698132, 1.39626]) #for grid plots
#phase_array = array([0.0, 0.628319, 1.25664])
#phase_array = array([1.88496, 2.51327, 3.14159])
#colorlist=linspace(0,1,len(phase_array))

#ticklabels=['0', '$\pi$', '$2 \pi$']
ticklabels=['0', '', '', '$2 \pi$']
#ticklabels=['0', '$\pi/3$', '$2 \pi/3$', '$\pi$', '$4 \pi/3$', '$5 \pi/3$', '$2 \pi$']
#ticklabels=['0', '$\pi/6$', '$\pi/3$', '$\pi/2$', '$2 \pi/3$']
ticklst=linspace(0,2*math.pi,len(ticklabels))

def calc_flux(p_now, drift_at_pos, diffusion_at_pos, flux_array):
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

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            input_file_name = (target_dir + "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/190610_Extra_measurements/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                       
            integrate_flux_X = empty(phase_array.size)
            integrate_flux_Y = empty(phase_array.size)
            integrate_power_X = empty(phase_array.size)
            integrate_power_Y = empty(phase_array.size)
            efficiency_ratio = empty(phase_array.size)
    
            for Ecouple in Ecouple_array:
                for ii, phase_shift in enumerate(phase_array):
        
                    print("Calculating flux for " + f"psi_2 = {psi_2}, psi_1 = {psi_1}, " + f"Ecouple = {Ecouple}, phase = {phase_shift}")
                    flux_array = empty((2,N,N))
                    #print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift))
                    data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift), usecols=(0,3,4,5,6,7,8))
                    prob_ss_array = data_array[:,0].reshape((N,N))
                    drift_at_pos = data_array[:,1:3].T.reshape((2,N,N))
                    diffusion_at_pos = data_array[:,3:].T.reshape((4,N,N))
    
                    calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array)
    
                    flux_array = asarray(flux_array)/(dx*dx)
    
                    integrate_flux_X[ii] = (1./(2*pi))*trapz(trapz(flux_array[0,...], dx=dx, axis=1), dx=dx)
                    integrate_flux_Y[ii] = (1./(2*pi))*trapz(trapz(flux_array[1,...], dx=dx, axis=0), dx=dx)
    
                    #print(sum(integrate_flux_Y))
                    integrate_power_X[ii] = integrate_flux_X[ii]*psi_1
                    integrate_power_Y[ii] = integrate_flux_Y[ii]*psi_2
                if (abs(psi_1) <= abs(psi_2)):
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
    
def plot_efficiency_grid():
    output_file_name = ("efficiency_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,5))
                    print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    eff_array = data_array[:,1]
                    if i==j:
                        eff_array = [-1,-1,-1,-1,-1,-1]
    
                    axarr[i,j].plot(phase_array,eff_array, color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')    
    #plt.legend(Ecouple_array, title="Ecouple")  
    plt.ylim(-0.3,1.0)  
    f.text(0.5, 0.04, '$\phi$', ha='center')
    f.text(0.04, 0.5, '$\eta$', va='center', rotation='vertical')
    plt.xticks(ticklst, ticklabels)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
        
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
    output_file_name = ("Twopisweep/master_output_dir/flux_XY_plot_Ecouple_0.0_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("Twopisweep/master_output_dir/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    #print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    #print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    flux_x_array = data_array[:,1]
                    flux_y_array = data_array[:,2]

                    ax.plot(phase_array, flux_x_array, 'o', markersize=4, color=plt.cm.cool(colorlist[ii]), label=f'{Ecouple}')
                    ax.plot(phase_array, flux_y_array, 'v', markersize=4, color=plt.cm.cool(colorlist[ii]))
                except OSError:
                    print('Missing file')   
            
            ##infinite coupling data
            # input_file_name = ("Twopisweep/master_output_dir/processed_data/Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
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
            
def plot_flux_Ecouple_single():
    output_file_name = ("190528_Phase_offset_results/master_output_dir/" + "flux_Ecouple_XY_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_log_grid_.pdf")
   
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            for i, phase in enumerate(phase_array):
                flux_x_array=[]
                flux_y_array=[]
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = ("190528_Phase_offset_results/master_output_dir/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                        flux_x = data_array[i,1]
                        flux_y = data_array[i,2]
                        flux_x_array.append(flux_x)
                        flux_y_array.append(flux_y)
                    except OSError:
                        print('Missing file')    
                plt.plot(Ecouple_array, flux_x_array, 'o', color=plt.cm.cool(colorlist[i]), label=f'{phase}')
                plt.plot(Ecouple_array, flux_y_array, 'v', color=plt.cm.cool(colorlist[i]))        
            
                # flux_x_array=[]
                # flux_y_array=[]
                # for ii, Ecouple in enumerate(Ecouple_array2):
                #     input_file_name = ("190530_Twopisweep/master_output_dir/" + "processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                #     try:
                #         data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                #         #print('Ecouple=%f'%Ecouple)
                #         #phase_array = data_array[:,0]
                #         flux_x = data_array[i,1]
                #         flux_y = data_array[i,2]
                #         flux_x_array.append(flux_x)
                #         flux_y_array.append(flux_y)
                #     except OSError:
                #         print('Missing file')
                # plt.plot(Ecouple_array2, flux_x_array, 'o', color=plt.cm.cool(colorlist[i]))
                # plt.plot(Ecouple_array2, flux_y_array, 'v', color=plt.cm.cool(colorlist[i]))
            plt.legend(title="$\phi$", loc='upper right')  
            plt.xlabel('$E_{couple}$')
            plt.ylabel('Flux')
            plt.xscale('log')
            plt.grid(True, which='both')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))                

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

        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array)
        #calc_flux_func(prob_ss_array, drift_at_pos[0]*(-1.0)*gamma, drift_at_pos[1]*(-1.0)*gamma, flux_array, m1, m2, gamma, beta, N, dx)

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

    calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array)

    flux_array = asarray(flux_array)/(dx*dx)

    integrate_flux_X = trapz(flux_array[0,...], dx=dx, axis=1)
    integrate_flux_Y = trapz(flux_array[1,...], dx=dx, axis=0)
    
    plt.plot(positions, integrate_flux_X)
    #plt.ylim((0,None))
    plt.show()

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

        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array)
        #calc_flux_func(prob_ss_array, drift_at_pos[0]*(-1.0)*gamma, drift_at_pos[1]*(-1.0)*gamma, flux_array, m1, m2, gamma, beta, N, dx)

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
    psi_1=4.0
    psi_2=-2.0
    Fx_array=empty((N,N))
    Fy_array=empty((N,N))
    Fx=psi_1*positions
    Fy=psi_2*positions
    for k in range(0,N):
        Fx_array[k]=Fx
        Fy_array[:,k]=Fy
    
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    output_file_name = ("Energy_force_flux_grid_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_30_2_.pdf")
    
    for i, phase in enumerate(phase_selection):
        for j, Ecouple in enumerate(Ecouple_array):
            flux_array = empty((2,N,N))
            input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            
            try:
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,2,3,4,5,6,7,8))
                prob_ss_array = data_array[:,0].reshape((N,N))
                pot_array = data_array[:,1].reshape((N,N))
                drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
                diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
                
                im = axarr[i,j].contourf(positions, positions, (pot_array-Fx_array-Fy_array), 30, vmin=-24, vmax=32)#plot energy landscape
                #im = axarr[i,j].contourf(positions, positions, pot_array, vmin=0, vmax=20)#plot energy landscape

                calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array)
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

                axarr[i,j].quiver(positions[::int(N/M)], positions[::int(N/M)], fluxX, fluxY, units='xy', angles='xy', scale_units='xy', scale=0.00000001)
                axarr[i,j].set_aspect(aspect=1, adjustable='box-forced')

            except OSError:
                print('Missing file')

    f.text(0.45, 0.04, '$X$', ha='center')
    f.text(0.04, 0.5, '$Y$', va='center', rotation='vertical')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xticks(ticklst, ticklabels)
    plt.yticks(ticklst, ticklabels)
    f.subplots_adjust(right=0.8)
    cbar_ax=f.add_axes([0.85, 0.25, 0.03, 0.5])
    f.colorbar(im, cax=cbar_ax)
    f.subplots_adjust(wspace=0.01, hspace=0.1)
    plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
    target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/190530_Twopisweep/master_output_dir"
    os.chdir(target_dir)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2)) 
    #plt.show()    
    
def plot_prob_flux():

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            f,axarr=plt.subplots(3,6,sharex='all',sharey='all', figsize=(12,6))
            output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" + "190528_Phase_offset_results/master_output_dir/ProbSS_flux_grid_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_big_.pdf")
            
            ##determining the max. potential height in the grid of plots, and the max. flux in the whole grid, so that we can scale the colors and arrows by it
            input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            data_array = loadtxt(input_file_name.format(E0, 16.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=(0,2,3,4,5,6,7,8))
            prob_ss_array = data_array[:,0].reshape((N,N))
            maxprob=amax(prob_ss_array)
            drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
            diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
            flux_array = empty((2,N,N))
            calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array)
            flux_array = asarray(flux_array)/(dx*dx)
            flux_length_array = empty((N,N))
            flux_x_array = flux_array[0]
            flux_y_array = flux_array[1]
            flux_length_array = flux_x_array*flux_x_array + flux_y_array*flux_y_array
            maxflux=amax(flux_length_array)
            print(maxprob, math.sqrt(maxflux))
            
            #actually making subplots
            for i, Ecouple in enumerate(Ecouple_array):
                for j, phase in enumerate(phase_array):
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase))
                    flux_array = empty((2,N,N))
                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase), usecols=(0,2,3,4,5,6,7,8))
                        prob_ss_array = data_array[:,0].reshape((N,N))
                        #pot_array = data_array[:,1].reshape((N,N))
                        drift_at_pos = data_array[:,2:4].T.reshape((2,N,N))
                        diffusion_at_pos = data_array[:,4:].T.reshape((4,N,N))
                        im = axarr[i,j].contourf(positions, positions, prob_ss_array.T, vmin=0, vmax=maxprob, cmap=plt.cm.cool)#plot energy landscape  
                
                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array)
                        flux_array = asarray(flux_array)/(dx*dx)
                        flux_x_array = (flux_array[0])
                        flux_y_array = (flux_array[1])
    
                        #select fewer arrows to draw
                        M = 18 #number of arrows in a row/ column, preferably a number such that N/M is an integer.
                        fluxX = empty((M, M))
                        fluxY = empty((M, M))
                        for k in range(M):
                            fluxX[k] = flux_x_array[int(N/M)*k, ::int(N/M)]
                            fluxY[k] = flux_y_array[int(N/M)*k, ::int(N/M)]
                        fluxzeros = zeros((M, M))    

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
            #f.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
            plt.close()
    
if __name__ == "__main__":
    target_dir="/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190520_phaseoffset/" #raw data 
    #target_dir="/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/2019-05-14/" #raw data
    #target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/" #processed data
    os.chdir(target_dir)
    #flux_power_efficiency()
    #plot_flux_single()
    #plot_flux_Ecouple_single()
    #plot_flux_contour()
    #plot_flux_grid()
    #plot_power_single()
    #plot_power_grid()
    #plot_efficiency_single()
    #plot_efficiency_grid()
    #plot_energy_flux()
    #plot_energy_flux_grid()
    plot_prob_flux()