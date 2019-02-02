from math import pi
from numpy import (
    array, arange, empty, finfo, pi as npi, log, true_divide, asarray,
    empty, zeros, linspace
    )
from time import time
from datetime import datetime
from os.path import isfile

from fpe_1d import (
    launchpad_reference
    )

def get_params():

    # discretization parameters
    dt = 0.001  # time discretization. Keep this number low
    N = 360  # inverse space discretization. Keep this number high!

    # time-specific parameters
    total_time = 10.0

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass

    A = 3.0 # energy scale of system X
    H = 0.0 # force on system X by chemical bath B1

    return (
        dt, N,
        gamma, beta, m,
        A, H
        )

def save_data_reference(
    A, H, p_now, p_equil, potential_at_pos, force_at_pos, N
    ):

    target_dir = './master_output_dir/'
    data_filename = '/reference_A_{0}_F_{1}_outfile.dat'.format(A, H)
    data_total_path = target_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            dfile.write(
                '{0:.15e}'.format(p_now[i])
                + '\t' + '{0:.15e}'.format(p_equil[i])
                + '\t' + '{0:.15e}'.format(potential_at_pos[i])
                + '\t' + '{0:.15e}'.format(force_at_pos[i])
                + '\n'
            )

def main():

    # unload parameters
    [dt, N, gamma, beta, m, A, H] = get_params()

    # calculate derived discretization parameters
    dx = (2*pi) / N  # space discretization: total distance / number of points

    # provide CSL criteria to make sure simulation doesn't blow up
    if A == 0.0:
        time_check = 100000000.0
    else:
        time_check = dx*m*gamma / (3*A)

    if dt > time_check:
        print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
        exit(1)

    # how many time update steps before checking for steady state convergence
    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print("Number of times before check = {0}".format(check_step))

    prob = zeros(N)
    p_now = zeros(N)
    p_last = zeros(N)
    p_last_ref = zeros(N)
    positions = linspace(0, (2*pi)-dx, N)
    potential_at_pos = zeros(N)
    force_at_pos = zeros(N)

    print("{} Launching reference simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    t0 = time()
    launchpad_reference(
        positions,
        prob,
        p_now, p_last, p_last_ref,
        potential_at_pos,
        force_at_pos,
        N, dx, check_step, A, H, dt, m, beta, gamma
    )
    t1 = time()
    comp_time = t1-t0
    print("{} Reference simulation done!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    print("{} Processing data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    # recast everything into a numpy array
    p_now = asarray(p_now)
    p_equil = asarray(prob)
    potential_at_pos = asarray(potential_at_pos)
    force_at_pos = asarray(force_at_pos)

    # for checking normalization
    check_sum = p_now.sum()

    # checks to make sure nothing went weird
    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    assert ((check_sum - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    print("{} Processing finished!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    # write to file
    print("{} Saving data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    save_data_reference(
        A, H, p_now, p_equil, potential_at_pos, force_at_pos, N
        )
    print("{} Saving completed!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    print("Exiting...")

if __name__ == "__main__":
    main()
