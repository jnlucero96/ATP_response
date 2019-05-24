import os
import glob
import re
from numpy import array, linspace, empty, loadtxt, asarray, pi
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
psi1_array = array([1.0, 2.0, 4.0])
psi2_array = array([-1.0, -2.0, -4.0])
#psi1_array = array([8.0])
#psi2_array = array([0.0])
Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
phase_array = array([0.0, 0.628319, 1.25664, 1.88496, 2.51327, 3.14159])
colorlst=['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']

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
            input_file_name = ("reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            output_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
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
    output_file_name = ("power_in_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,3,4))
                    print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    power_x_array = data_array[:,1]
                    power_y_array = data_array[:,2]
    
                    axarr[i,j].plot(phase_array,power_x_array, color=colorlst[ii])
                except OSError:
                    print('Missing file')    
            #plt.legend(Ecouple_array, title="Ecouple")    
            f.text(0.5, 0.04, '$\phi$', ha='center')
            f.text(0.04, 0.5, 'Input power', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))

def plot_power_single():
    output_file_name = ("power_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()    
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,4))
                    print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    power_y_array = data_array[:,1]

                    plt.plot(phase_array, power_y_array, color=colorlst[ii])
                except OSError:
                    print('Missing file')    
            plt.legend(Ecouple_array, title="Ecouple", loc='upper left')    
            #f.text(0.5, 0.04, '$\phi$', ha='center')
            #f.text(0.04, 0.5, 'Output power', va='center', rotation='vertical')
            plt.xlabel('$\phi$')
            plt.ylabel('Output power')
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
    
def plot_efficiency_single():
    output_file_name = ("efficiency_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()    
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,5))
                    print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    eff_array = data_array[:,1]

                    plt.plot(phase_array, eff_array, color=colorlst[ii])
                except OSError:
                    print('Missing file')    
            plt.legend(Ecouple_array, title="Ecouple", loc='upper left')    
            plt.xlabel('$\phi$')
            plt.ylabel('$\eta$')
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))
    
def plot_efficiency_grid():
    output_file_name = ("efficiency_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,5))
                    print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    eff_array = data_array[:,1]
                    if i==j:
                        eff_array = [-1,-1,-1,-1,-1,-1]
    
                    axarr[i,j].plot(phase_array,eff_array, color=colorlst[ii])
                except OSError:
                    print('Missing file')    
            #plt.legend(Ecouple_array, title="Ecouple")  
            plt.ylim(-0.3,1.0)  
            f.text(0.5, 0.04, '$\phi$', ha='center')
            f.text(0.04, 0.5, '$\eta$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
        
def plot_flux_grid():
    output_file_name = ("flux_x_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f,axarr=plt.subplots(3,3,sharex='all',sharey='all')
    for i, psi_1 in enumerate(psi1_array):
        for j, psi_2 in enumerate(psi2_array):
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    flux_x_array = data_array[:,1]
                    flux_y_array = data_array[:,2]
    
                    axarr[i,j].plot(phase_array, flux_y_array, color=colorlst[ii])
                except OSError:
                    print('Missing file')    
            #plt.legend(Ecouple_array, title="Ecouple")    
            f.text(0.5, 0.04, '$\phi$', ha='center')
            f.text(0.04, 0.5, '$J_x$', va='center', rotation='vertical')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))
        
def plot_flux_single():
    output_file_name = ("flux_X_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    
    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            for ii, Ecouple in enumerate(Ecouple_array):
                input_file_name = ("flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(0,1,2))
                    print('Ecouple=%f'%Ecouple)
                    phase_array = data_array[:,0]
                    flux_x_array = data_array[:,1]
                    flux_y_array = data_array[:,2]

                    plt.plot(phase_array, flux_x_array, color=colorlst[ii])
                except OSError:
                    print('Missing file')    
            plt.legend(Ecouple_array, title="Ecouple", loc='upper left')    
            #f.text(0.5, 0.04, '$\phi$', ha='center')
            #f.text(0.04, 0.5, 'Output power', va='center', rotation='vertical')
            #plt.ylim((0,None))
            plt.xlabel('$\phi$')
            plt.ylabel('Flux X')
            plt.grid()
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

if __name__ == "__main__":
    target_dir="/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/fokker_planck/working_directory_cython/working_directory_cython/working_directory_cython/master_output_dir"
    os.chdir(target_dir)
    #flux_power_efficiency()
    plot_flux_single()
    #plot_flux_contour()
    #plot_flux_grid()
    #plot_power_single()
    #plot_power_grid()
    #plot_efficiency_single()
    #plot_efficiency_grid()