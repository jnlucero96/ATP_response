#!/anaconda3/bin/python
from math import pi
from numpy import (
    array, arange, empty, finfo, pi as npi, log, true_divide, asarray,
    empty, zeros, linspace
    )
from time import time
from datetime import datetime
from os.path import isfile

from fpe import launchpad_coupled

def get_ID_value(filename=".varstore"):
    try:
        with open(filename, "r+") as rfile:
            val = int(rfile.read()) + 1
    except FileNotFoundError:
        with open(filename, "w") as ofile:
            val = 1
    with open(filename, "w") as wfile:
        wfile.write(str(val))
    return val

def get_params():

    ID = get_ID_value()
    # discretization parameters
    dt = 0.001  # time discretization. Keep this number low
    N = 1800  # inverse space discretization. Keep this number high!

    # time-specific parameters
    total_time = 0.5

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass
    Ax = 10.0 # energy scale of system X
    Axy = 0.0 # energy scale of coupling between systems X and Y
    Ay = 0.0 # energy scale of system Y
    H = 10.0 # force on system X by chemical bath B1
    A = 0.0 # force on system Y by chemical bath B2

    # initialization condition: equilibrium , uniform, or steady
    initialize_from = 'steady'

    # save data or not
    save = True

    return (
        ID, dt, N, total_time,
        gamma, beta, m,
        Ax, Ay, Axy, H, A,
        initialize_from, save
        )

def save_data(
    ID, total_time, Ax, Axy, Ay, H, A,
    p_now, check_sum, p_equil, distance, rel_entropy,
    flux, mean_space_flux, mean_space_time_flux,
    potential_at_pos, force1_at_pos, force2_at_pos,
    N, comp_time
    ):

    target_dir = './output_dir/'
    data_filename = '/ID_{0}_outfile.dat'.format(ID)
    data_total_path = target_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            for j in range(N):
                dfile.write(
                    '{0:.15e}'.format(p_now[i, j])
                    + '\t' + '{0:.15e}'.format(p_equil[i, j])
                    + '\t' + '{0:.15e}'.format(flux[0, i, j])
                    + '\t' + '{0:.15e}'.format(flux[1, i, j])
                    + '\t' + '{0:.15e}'.format(potential_at_pos[i, j])
                    + '\t' + '{0:.15e}'.format(force1_at_pos[i, j])
                    + '\t' + '{0:.15e}'.format(force2_at_pos[i, j])
                    + '\n'
                )

    summary_filename = '/summary.dat'
    summary_total_path = target_dir + summary_filename

    with open(summary_total_path, 'a') as sfile:

        sfile.write(
            "Simulation ID: {0}".format(ID) + '\n'
            + "Ax = {0}, Axy = {1}, Ay = {2}".format(Ax, Axy, Ay) + '\n'
            + "H = {0}, A = {1}".format(H, A) + '\n'
            + "Real T = {0}, Simulation T = {1}, N = {2}, Normalization = {3}".format(comp_time, total_time, N, check_sum) + '\n'
            + "Total Variation Distance = {0}, D(P||pi_eq) = {1}".format(distance, rel_entropy) + '\n'
            + "<J1> = {0}, <J2> = {1}, <J> = {2}".format(mean_space_flux[0], mean_space_flux[1], mean_space_flux.mean()) + '\n'
            + "<<J1>> = {0}, <<J2>> = {1}, <<J>>= {2}".format(mean_space_time_flux[0], mean_space_time_flux[1], mean_space_time_flux.mean()) + '\n\n'
            )

def main():

    # unload parameters
    [
        ID, dt, N, total_time, gamma, beta, m, Ax, Ay, Axy, H, A, initialize_from, save
        ] = get_params()

    # calculate derived discretization parameters
    dx = (2*pi) / N  # space discretization: total distance / number of points

    # provide CSL criteria to make sure simulation doesn't blow up
    if Ax == 0.0 and Ay == 0.0:
        time_check = 100000000.0
    else:
        time_check = dx*m*gamma / (3*(Ax + Ay))

    if dt > time_check:
        print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
        exit(1)

    if initialize_from.lower() == "equilibrium":
        steady_state_var = 0
    elif initialize_from.lower() == "uniform":
        steady_state_var = 1
    elif initialize_from.lower() == "steady":
        steady_state_var = 2
    else:
        print("Initialization condition not understood! Aborting...")
        exit(1)

    print("Initialization condition: " + initialize_from)

    prob = zeros((N, N))
    p_now = zeros((N, N))
    p_last = zeros((N, N))
    p_last_ref = zeros((N, N))
    flux = zeros((2, N, N))  # array to keep
    positions = linspace(0, (2*pi)-dx, N)
    potential_at_pos = zeros((N, N))
    force1_at_pos = zeros((N, N))
    force2_at_pos = zeros((N, N))

    # count number of times the primary loops correspond to the desired
    num_loops = int((total_time+dt)//dt)

    print("Number of times around loop = {0}".format(num_loops))
    print("{} Launching simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    t0 = time()
    work, heat = launchpad_coupled(
        positions, prob, p_now, p_last, p_last_ref, flux,
        potential_at_pos, force1_at_pos, force2_at_pos,
        N, num_loops,
        dx, time_check, steady_state_var,
        Ax, Axy, Ay, H, A,
        dt, m, beta, gamma
    )
    t1 = time()
    comp_time = t1-t0
    print("{} Simulation done!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    print("{} Processing data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    # recast everything into a numpy array
    flux = asarray(flux)
    p_now = asarray(p_now)
    p_equil = asarray(prob)
    positions = asarray(positions)
    potential_at_pos = asarray(potential_at_pos)
    force1_at_pos = asarray(force1_at_pos)
    force2_at_pos = asarray(force2_at_pos)

    # average over the positions
    mean_space_flux = flux.mean(axis=(1, 2))
    mean_space_time_flux = mean_space_flux / float(num_loops)

    # for checking normalization
    check_sum = p_now.sum(axis=None)

    # checks to make sure nothing went weird
    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    assert ((check_sum - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    distance = 0.5*(p_equil - p_now).__abs__().sum(axis=None)
    rel_entropy = p_now.dot(log(p_now / p_equil)).sum(axis=None)

    print("{} Processing finished!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    # write to file or to stdout
    if save:
        print("{} Saving data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        save_data(
            ID, total_time, Ax, Axy, Ay, H, A,
            p_now, check_sum, p_equil, distance, rel_entropy,
            flux, mean_space_flux, mean_space_time_flux,
            potential_at_pos, force1_at_pos, force2_at_pos,
            N, comp_time
            )
        print("{} Saving completed!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    else:
        print("="*10 + " Simulation Summary " + "="*10)
        print("Simulation ID: {0}".format(ID))
        print(
            "Real T = {0}, Simulation T = {1}, N = {2}, Normalization = {3}".format(comp_time, total_time, N, check_sum)
            )
        print("H = {0}, A = {1}".format(H, A))
        print(
            "Total Variation Distance = {0}, D(P||pi_eq) = {1}".format(distance, rel_entropy)
            )
        print("<J1> = {0}, <J2> = {1}, <J> = {2}".format(mean_space_flux[0], mean_space_flux[1], mean_space_flux.mean()))
        print("<<J1>> = {0}, <<J2>> = {1}, <<J>>= {2}".format(mean_space_time_flux[0], mean_space_time_flux[1], mean_space_time_flux.mean()))
        print("="*40)

    print("Exiting...")

if __name__ == "__main__":
    main()
