from math import pi
from numpy import (
    array, arange, empty, finfo, pi as npi, log, true_divide, asarray,
    empty, zeros, linspace
    )
from time import time
from datetime import datetime
from os.path import isfile
from scipy.integrate import trapz

from fpe_1d import launchpad_reference
from utilities_1d import calc_flux_1d

def get_params():

    # discretization parameters
    dt = 0.001  # time discretization. Keep this number low
    N = 360  # inverse space discretization. Keep this number high!

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass

    A = 4.0 # energy scale of system

    H = 8.0 # force on system by chemical bath B1
    atp = -2.0 # force on system by chemical bath B2
    overall = 6.0

    num_minima = 3.0

    return (
        dt, N,
        gamma, beta, m,
        A, H, atp, overall, num_minima
        )

def save_data_reference(
    target_dir,
    A, H, atp, num_minima, p_now, flux_array, p_equil,
    potential_at_pos, force_at_pos, N
    ):

    # target_dir = './master_output_dir/'
    data_filename = '/reference_E0_{0}_F_Hplus_{1}_F_atp_{2}_minima_{3}_outfile.dat'.format(A, H, atp, num_minima)
    data_total_path = target_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            dfile.write(
                '{0:.15e}'.format(p_now[i])
                + '\t' + '{0:.15e}'.format(flux_array[i])
                + '\t' + '{0:.15e}'.format(p_equil[i])
                + '\t' + '{0:.15e}'.format(potential_at_pos[i])
                + '\t' + '{0:.15e}'.format(force_at_pos[i])
                + '\n'
            )

def main(target_dir):

    # unload parameters
    [dt, N, gamma, beta, m, A, H, atp, overall, num_minima] = get_params()

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
    positions = linspace(0.0, (2*pi)-dx, N)
    potential_at_pos = zeros(N)
    force_at_pos = zeros(N)
    flux_array = zeros(N)

    print("{} Launching reference simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    launchpad_reference(
        positions,
        prob,
        p_now, p_last, p_last_ref,
        potential_at_pos,
        force_at_pos,
        N, dx, check_step, A, H, atp, overall,
        num_minima, dt, m, beta, gamma
    )
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

    print("{} Calculating flux...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    calc_flux_1d(
        positions, p_now, force_at_pos, flux_array,
        m, gamma, beta, N, dx, dt
        )

    print("{} Processing finished!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    # write to file
    print("{} Saving data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    save_data_reference(
        target_dir,
        A, H, atp, num_minima, p_now, flux_array, p_equil,
        potential_at_pos, force_at_pos, N
        )
    print("{} Saving completed!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    print("Exiting...")

if __name__ == "__main__":
    target_dir = "/Users/jlucero/data_to_not_upload/2019-03-20/"
    main(target_dir)
