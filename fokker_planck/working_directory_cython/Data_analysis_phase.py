import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi, meshgrid
import math
import matplotlib.pyplot as plt
from scipy.integrate import trapz

N=360
dx=2*math.pi/N
#gamma=1000
#m1=1
#m2=1
#beta=1
positions=linspace(0,2*math.pi-dx,N)
E0=2.0
E1=2.0
num_minima1=3.0
num_minima2=3.0
psi1_array = array([4.0])
psi2_array = array([-2.0, -1.0])
#psi1_array = array([0.0])
#psi2_array = array([0.0])
Ecouple_array = array([0.0])
#Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
#phase_array = array([0.0, 0.628319, 1.25664, 1.88496, 2.51327, 3.14159])
#phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944, 2.44346, 2.79253, 3.14159, 3.49066, 3.83972, 4.18879, 4.53786, 4.88692, 5.23599, 5.58505, 5.93412, 6.28319])
phase_selection = array([0.0, 0.698132, 1.39626, 2.0944, 2.79253])
colorlist=linspace(0,1,4)
ticklst=linspace(0,2*math.pi,7)
#ticklabels=['0', '$\pi/6$', '$\pi/3$', '$\pi/2$', '$2 \pi/3$']
ticklabels=['0', '$\pi/3$', '$2 \pi/3$', '$\pi$', '$4 \pi/3$', '$5 \pi/3$', '$2 \pi$']

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
            output_file_name = ("Twopisweep/master_output_dir/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
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
    output_file_name = ("Twopisweep/master_output_dir/flux_XY_plot_Ecouple_0.0_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_try.pdf")
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
    output_file_name = ("Twopisweep/master_output_dir/flux_Ecouple_XY_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
   
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            ax=plt.subplot(111)
            ax.axhline(0, color='black', linewidth=2)
            for i, phase in enumerate(phase_selection):
                flux_x_array=[]
                flux_y_array=[]
                print('Figure for psi1=%f, psi2=%f' % (psi_1, psi_2))
                for ii, Ecouple in enumerate(Ecouple_array):
                    input_file_name = ("Twopisweep/master_output_dir/processed_data/flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                        #print('Ecouple=%f'%Ecouple)
                        #phase_array = data_array[:,0]
                        flux_x = data_array[i,1]
                        flux_y = data_array[i,2]
                        flux_x_array.append(flux_x)
                        flux_y_array.append(flux_y)
                    except OSError:
                        print('Missing file')    
                plt.plot(Ecouple_array, flux_x_array, color=plt.cm.cool(colorlist[i]), ls='--')
                plt.plot(Ecouple_array, flux_y_array, color=plt.cm.cool(colorlist[i]), label=f'{phase}')        
            #plt.legend(Ecouple_array, title="Ecouple", loc='upper left')    
            plt.legend(title="$\phi$", loc='upper right')  
            #f.text(0.5, 0.04, '$\phi$', ha='center')
            #f.text(0.04, 0.5, 'Output power', va='center', rotation='vertical')
            #plt.ylim((0,None))
            plt.xlabel('$E_{couple}$')
            plt.ylabel('Flux')
            #plt.xticks(ticklst, ticklabels)
            #plt.grid()
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
    
if __name__ == "__main__":
    #target_dir="/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190530_phaseoffset_twopi"
    #target_dir="/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/2019-05-14/"
    target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/"
    os.chdir(target_dir)
    #flux_power_efficiency()
    #plot_flux_single()
    #plot_flux_Ecouple_single()
    #plot_flux_contour()
    #plot_flux_grid()
    plot_power_single()
    #plot_power_grid()
    #plot_efficiency_single()
    #plot_efficiency_grid()
    #plot_energy_flux()
    
    
    
# def calc_flux_func(p_now,force1_at_pos, force2_at_pos,flux_array,m1, m2, gamma, beta, N, dx):
#
#     # explicit update of the corners
#     # first component
#     flux_array[0, 0, 0] = (-1.0)*(
#         (force1_at_pos[0, 0]*p_now[0, 0])/(gamma*m1)
#         + (p_now[1, 0] - p_now[N-1, 0])/(beta*gamma*m1*2*dx)
#         )
#     flux_array[0, 0, N-1] = (-1.0)*(
#         (force1_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m1)
#         + (p_now[1, N-1] - p_now[N-1, N-1])/(beta*gamma*m1*2*dx)
#         )
#     flux_array[0, N-1, 0] = (-1.0)*(
#         (force1_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m1)
#         + (p_now[0, 0] - p_now[N-2, 0])/(beta*gamma*m1*2*dx)
#         )
#     flux_array[0, N-1, N-1] = (-1.0)*(
#         (force1_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m1)
#         + (p_now[0, N-1] - p_now[N-2, N-1])/(beta*gamma*m1*2*dx)
#         )
#
#     # second component
#     flux_array[1, 0, 0] = (-1.0)*(
#         (force2_at_pos[0, 0]*p_now[0, 0])/(gamma*m2)
#         + (p_now[0, 1] - p_now[0, N-1])/(beta*gamma*m2*2*dx)
#         )
#     flux_array[1, 0, N-1] = (-1.0)*(
#         (force2_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m2)
#         + (p_now[0, 0] - p_now[0, N-2])/(beta*gamma*m2*2*dx)
#         )
#     flux_array[1, N-1, 0] = (-1.0)*(
#         (force2_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m2)
#         + (p_now[N-1, 1] - p_now[N-1, N-1])/(beta*gamma*m2*2*dx)
#         )
#     flux_array[1, N-1, N-1] = (-1.0)*(
#         (force2_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m2)
#         + (p_now[N-1, 0] - p_now[N-1, N-2])/(beta*gamma*m2*2*dx)
#         )
#
#     # for points with well defined neighbours
#     for i in range(1, N-1):
#         # explicitly update for edges not corners
#         # first component
#         flux_array[0, 0, i] = (-1.0)*(
#             (force1_at_pos[0, i]*p_now[0, i])/(gamma*m1)
#             + (p_now[1, i] - p_now[N-1, i])/(beta*gamma*m1*2*dx)
#         )
#         flux_array[0, i, 0] = (-1.0)*(
#             (force1_at_pos[i, 0]*p_now[i, 0])/(gamma*m1)
#             + (p_now[i+1, 0]- p_now[i-1, 0])/(beta*gamma*m1*2*dx)
#         )
#
#         # second component
#         flux_array[1, 0, i] = (-1.0)*(
#             (force2_at_pos[0, i]*p_now[0, i])/(gamma*m2)
#             + (p_now[0, i+1] - p_now[0, i-1])/(beta*gamma*m2*2*dx)
#             )
#         flux_array[1, i, 0] = (-1.0)*(
#             (force2_at_pos[i, 0]*p_now[i, 0])/(gamma*m2)
#             + (p_now[i, 1] - p_now[i, N-1])/(beta*gamma*m2*2*dx)
#             )
#
#         for j in range(1, N-1):
#             # first component
#             flux_array[0, i, j] = (-1.0)*(
#                 (force1_at_pos[i, j]*p_now[i, j])/(gamma*m1)
#                 + (p_now[i+1, j] - p_now[i-1, j])/(beta*gamma*m1*2*dx)
#                 )
#             # second component
#             flux_array[1, i, j] = (-1.0)*(
#                 (force2_at_pos[i, j]*p_now[i, j])/(gamma*m2)
#                 + (p_now[i, j+1] - p_now[i, j-1])/(beta*gamma*m2*2*dx)
#                 )
#
#         # update rest of edges not corners
#         # first component
#         flux_array[0, N-1, i] = (-1.0)*(
#             (force1_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m1)
#             + (p_now[0, i] - p_now[N-2, i])/(beta*gamma*m1*2*dx)
#             )
#         flux_array[0, i, N-1] = (-1.0)*(
#             (force1_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m1)
#             + (p_now[i+1, N-1] - p_now[i-1, N-1])/(beta*gamma*m1*2*dx)
#             )
#
#         # second component
#         flux_array[1, N-1, i] = (-1.0)*(
#             (force2_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m2)
#             + (p_now[N-1, i+1] - p_now[N-1, i-1])/(beta*gamma*m2*2*dx)
#             )
#         flux_array[1, i, N-1] = (-1.0)*(
#             (force2_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m2)
#             + (p_now[i, 0] - p_now[i, N-2])/(beta*gamma*m2*2*dx)
#             )