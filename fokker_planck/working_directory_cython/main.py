from math import pi
from numpy import (
    array, arange, empty, finfo, pi as npi, log, true_divide, asarray,
    empty, zeros, linspace
    )
from time import time

from fpe import launchpad_coupled

def get_params():

    ID = 5
    # discretization parameters
    dt = 0.005  # time discretization. Keep this number low
    N = 360  # inverse space discretization. Keep this number high!

    # time-specific parameters
    total_time = 1000.0

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass
    Ax = 10.0 # energy scale of system X
    Axy = 0.0 # energy scale of coupling between systems X and Y
    Ay = 0.0 # energy scale of system Y
    H = 10.0 # force on system X by chemical bath B1
    A = 0.0 # force on system Y by chemical bath B2

    # initialization condition
    initialize_from = 'uniform'

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
            + "<<J1>> = {0}, <<J2>> = {1}, <<J> >= {2}".format(mean_space_time_flux[0], mean_space_time_flux[1], mean_space_time_flux.mean()) + '\n\n'
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

    if initialize_from.lower() == "equilibrium":
        steady_state_var = 0
    elif initialize_from.lower() == "uniform":
        steady_state_var = 1
    else:
        print("Initialization condition not understood. Aborting...")
        exit(1)

    prob = zeros((N, N))
    p_now = zeros((N, N))
    p_last = zeros((N, N))
    p_last_100 = zeros((N, N))
    flux = zeros((2, N, N))  # array to keep
    positions = linspace(0, (2*pi)-dx, N)

    # count number of times the primary loops correspond to the desired
    #
    num_loops = int((total_time+dt)//dt)
    num_loops_by_two = int(num_loops // 2) # for calculating the flux

    print("Number of times around loop = {0}".format(num_loops))
    print("Launching!")
    t0 = time()
    work, heat = launchpad_coupled(
        prob, p_now, p_last, p_last_100, flux, positions,
        N, num_loops, num_loops_by_two,
        dx, time_check, steady_state_var,
        Ax, Axy, Ay, H, A,
        dt, m, beta, gamma
    )
    t1 = time()
    comp_time = t1-t0
    print("Finished! Processing data now...")

    # recast everything into a numpy array
    flux = asarray(flux)
    p_now = asarray(p_now)
    p_equil = asarray(prob)
    positions = asarray(positions)

    # average over the positions
    mean_space_flux = flux.mean(axis=(1, 2))
    mean_space_time_flux = mean_space_flux / float(num_loops_by_two)

    # for checking normalization
    check_sum = p_now.sum(axis=None)

    # checks to make sure nothing went weird
    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    assert ((check_sum - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    distance = 0.5*(p_equil - p_now).__abs__().sum(axis=None)
    rel_entropy = p_now.dot(log(p_now / p_equil)).sum(axis=None)

    print("Processing finished!")

    # write to file or to stdout
    if save:
        save_data(
            ID, total_time, Ax, Axy, Ay, H, A,
            p_now, check_sum, p_equil, distance, rel_entropy,
            flux, mean_space_flux, mean_space_time_flux,
            N, comp_time
            )
    else:
        print("Simulation ID: {0}".format(ID))
        print(
            "Real T = {0}, Simulation T = {1}, N = {2}, Normalization = {3}".format(comp_time, total_time, N, check_sum)
            )
        print("H = {0}, A = {1}".format(H, A))
        print(
            "Total Variation Distance = {0}, D(P||pi_eq) = {1}".format(distance, rel_entropy)
            )
        print("<J1> = {0}, <J2> = {1}, <J> = {2}".format(mean_space_flux[0], mean_space_flux[1], mean_space_flux.mean()))
        print("<<J1>> = {0}, <<J2>> = {1}, <<J> >= {2}".format(mean_space_time_flux[0], mean_space_time_flux[1], mean_space_time_flux.mean()))
        print()

    print("Exiting...")

if __name__ == "__main__":
    main()
