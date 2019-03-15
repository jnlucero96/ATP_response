from math import pi
from numpy import (
    array, arange, empty, finfo, log, true_divide, asarray,
    empty, zeros, linspace
    )
from datetime import datetime

from fpe import launchpad_reference

def get_params():

    # discretization parameters
    dt = 0.001  # time discretization. Keep this number low
    N = 360  # inverse space discretization. Keep this number high!

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass

    E0 = 3.0 # energy scale of F0 sub-system
    Ecouple = 3.0 # energy scale of coupling between sub-systems F0 and F1
    E1 = 3.0 # energy scale of F1 sub-system
    F_Hplus = 3.0 #  energy INTO (positive) F0 sub-system by H+ chemical bath
    F_atp = 3.0 # energy INTO (positive) F1 sub-system by ATP chemical bath

    num_minima = 3.0 # number of minima in the potential
    phase_shift = 0.0 # how much sub-systems are offset from one another

    return (
        dt, N,
        gamma, beta, m, num_minima, phase_shift,
        E0, E1, Ecouple, F_Hplus, F_atp
        )

def save_data_reference(
    num_minima, phase_shift,
    E0, Ecouple, E1, F_Hplus, F_atp, p_now, p_equil,
    potential_at_pos, force1_at_pos, force2_at_pos,
    N
    ):

    target_dir = './master_output_dir/'
    data_filename = '/reference_E0_{0}_Ecouple_{1}_E1_{2}_F_Hplus_{3}_F_atp_{4}_minima_{5}_phase_{6}_outfile.dat'.format(E0, Ecouple, E1, F_Hplus, F_atp, num_minima, phase_shift)
    data_total_path = target_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            for j in range(N):
                dfile.write(
                    '{0:.15e}'.format(p_now[i, j])
                    + '\t' + '{0:.15e}'.format(p_equil[i, j])
                    + '\t' + '{0:.15e}'.format(potential_at_pos[i, j])
                    + '\t' + '{0:.15e}'.format(force1_at_pos[i, j])
                    + '\t' + '{0:.15e}'.format(force2_at_pos[i, j])
                    + '\n'
                )

def main():

    # unload parameters
    [
        dt, N, gamma, beta, m, num_minima, phase_shift, E0, E1, Ecouple,
        F_Hplus, F_atp
        ] = get_params()

    # calculate derived discretization parameters
    dx = (2*pi) / N  # space discretization: total distance / number of points

    # provide CSL criteria to make sure simulation doesn't blow up
    if E0 == 0.0 and E1 == 0.0:
        time_check = 100000000.0
    else:
        time_check = dx*m*gamma / (3*(E0 + E1))

    if dt > time_check:
        print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
        exit(1)

    # how many time update steps before checking for steady state convergence
    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print("Number of times before check = {0}".format(check_step))

    prob = zeros((N, N))
    p_now = zeros((N, N))
    p_last = zeros((N, N))
    p_last_ref = zeros((N, N))
    positions = linspace(0, (2*pi)-dx, N)
    potential_at_pos = zeros((N, N))
    force1_at_pos = zeros((N, N))
    force2_at_pos = zeros((N, N))

    print("{} Launching coupled simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    launchpad_reference(
        num_minima, phase_shift,
        positions, prob, p_now, p_last, p_last_ref,
        potential_at_pos, force1_at_pos, force2_at_pos,
        N, dx, check_step,
        E0, Ecouple, E1, F_Hplus, F_atp,
        dt, m, beta, gamma
    )
    print("{} Coupled simulation done!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    print("{} Processing data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    # recast everything into a numpy array
    p_now = asarray(p_now)
    p_equil = asarray(prob)
    potential_at_pos = asarray(potential_at_pos)
    force1_at_pos = asarray(force1_at_pos)
    force2_at_pos = asarray(force2_at_pos)

    # for checking normalization
    check_sum = p_now.sum(axis=None)

    # checks to make sure nothing went weird
    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    assert ((check_sum - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    print("{} Processing finished!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    # write to file
    print("{} Saving data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    save_data_reference(
        num_minima, phase_shift,
        E0, Ecouple, E1, F_Hplus, F_atp, p_now, p_equil,
        potential_at_pos, force1_at_pos, force2_at_pos, N
        )
    print("{} Saving completed!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    print("Exiting...")

if __name__ == "__main__":
    main()
