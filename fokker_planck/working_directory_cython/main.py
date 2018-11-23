from math import pi
from numpy import (
    array, arange, empty, finfo, pi as npi, log, true_divide, asarray,
    empty, zeros, linspace
    )
from time import time

from fpe import launchpad_coupled

def get_params():

    # discretization parameters
    dt = 0.005  # time discretization. Keep this number low
    N = 720  # inverse space discretization. Keep this number high!

    # time-specific parameters
    cycles = 1.0  # Max number of cycles
    period = 1.0  # How long a cycle takes

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass
    Ax = 10.0 # energy scale of system X
    Axy = 0.0 # energy scale of coupling between systems X and Y
    Ay = 0.0 # energy scale of system Y
    H = 1000.0 # force on system X by chemical bath B1
    A = 0.0 # force on system Y by chemical bath B2

    # initialization conditions
    steady_state = False

    return (dt, N, cycles, period, gamma, beta, m, Ax, Ay, Axy, H, A, steady_state)

def save_data(times, amplitudes, trap_strengths, work_array, flux_array):

    rot_rate = 1./times

    target_dir = './output_dir/'
    for index1, A in enumerate(amplitudes):
        for index2, k in enumerate(trap_strengths):
            filename = '/output_file_k{0}_A{1}_final.dat'.format(k, A)
            with open(target_dir + filename, 'w') as ofile:
                for index3, t_and_f in enumerate(zip(times, rot_rate)):
                    t = t_and_f[0]
                    f = t_and_f[1]
                    ofile.write(
                        "{0}\t{1}\t{2}\t{3}\n".format(
                            t, f,
                            work_array[index1, index2, index3],
                            flux_array[index1, index2, index3]
                            )
                        )
                ofile.flush()

def main():

    # unload parameters
    [
        dt, N, cycles, period, gamma, beta, m, Ax, Ay, Axy, H, A, steady_state
        ] = get_params()

    # calculate derived discretization parameters
    dx = (2*pi) / N  # space discretization
    if Ax == 0.0 and Ay == 0.0:
        time_check = 100000000.0
    else:
        time_check = dx*m*gamma / (3*(Ax + Ay))

    if steady_state:
        steady_state_var = 1
    else:
        steady_state_var = 0

    prob = zeros((N, N))
    p_now = zeros((N, N))
    p_last = zeros((N, N))
    flux = zeros((2, N, N))  # array to keep
    positions = linspace(0, (2*pi)-dx, N)

    print("Number of times around loop = {0}".format((period*cycles+dt)/dt))
    print("Launching!")
    t0 = time()
    work, heat = launchpad_coupled(
        prob, p_now, p_last, flux, positions, N, dx, time_check,
        steady_state_var, cycles, period,
        Ax, Axy, Ay, H, A,
        dt, m, beta, gamma
    )
    t1 = time()
    total = t1-t0
    # p_equil, p_now_m = run_func(N)
    print("Finished! Processing data now...")
    flux = asarray(flux)
    mean_flux = flux.mean(axis=(1, 2))
    p_now = asarray(p_now)
    p_sum = p_now.sum(axis=None)
    p_equil = asarray(prob)
    positions = asarray(positions)

    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    assert ((p_now.sum(axis=None) - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized"

    # with open('T{0}_H{1}_A{2}_outfile.dat'.format(period, H, A), 'w') as ofile:
    #     for i in range(N):
    #         for j in range(N):
    #             ofile.write(
    #                 str(p_now[i, j])
    #                 + '\t' + str(flux[0, i, j])
    #                 + '\t' + str(flux[1, i, j])
    #                 + '\n'
    #             )

    distance = 0.5*(p_equil - p_now).__abs__().sum(axis=None)
    rel_entropy = p_now.dot(log(p_now / p_equil)).sum(axis=None)

    print("Processing finished!")

    print("Real T = {0}, Simulation T = {1}, N = {2}, Psum = {3}".format(
        total, period, N, p_sum))
    print("H = {0}, A = {1}".format(H, A))
    print("Total Variation Distance = {0}, D(P||pi_eq) = {1}".format(
        distance, rel_entropy))
    print("<J1> = {0}, <J2> = {1}".format(mean_flux[0], mean_flux[1]))

    print("Exiting...")

if __name__ == "__main__":
    main()
