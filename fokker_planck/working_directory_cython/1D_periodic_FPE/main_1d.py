#!/usr/bin/env python
from math import pi
from numpy import finfo, asarray, zeros, linspace
from datetime import datetime

from fpe_1d import launchpad_reference

def get_params():

    # discretization parameters
    dt = 1e-5  # time discretization. Keep this number low
    N = 360 # inverse space discretization. Keep this number high!

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass

    E = 8.0 # energy scale of system

    psi1 = 8.0 # force on system by chemical bath B1
    psi2 = -2.0 # force on system by chemical bath B2

    n = 3.0 # number of minima in system potential

    return ( dt, N, gamma, beta, m, E, psi1, psi2, n )

def save_data_reference(
    E, psi1, psi2, n, p_now, p_equil,
    potential_at_pos, drift_at_pos, diffusion_at_pos, N
    ):

    target_dir = './master_output_dir/'
    data_filename = f'/reference_E_{E}_psi1_{psi1}_psi2_{psi2}_n_{n}_outfile.dat'
    data_total_path = target_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            dfile.write(
                f"{p_now[i]:.15e}"
                + '\t' + f"{p_equil[i]:.15e}"
                + '\t' + f"{potential_at_pos[i]:.15e}"
                + '\t' + f"{drift_at_pos[i]:.15e}"
                + '\t' + f"{diffusion_at_pos[i]:.15e}"
                + '\n'
            )

def main():

    # unload parameters
    [ dt, N, gamma, beta, m, E, psi1, psi2, n ] = get_params()

    # calculate derived discretization parameters
    dx = (2*pi)/N  # space discretization: total distance / number of points

    # provide CSL criteria to make sure simulation doesn't blow up
    if E == 0.0:
        time_check = 100000000.0
    else:
        time_check = dx*m*gamma / (3*E)

    if dt > time_check:
        print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
        exit(1)

    # how many time update steps before checking for steady state convergence
    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print(f"Number of times before check = {check_step}")

    prob = zeros(N)
    p_now = zeros(N)
    p_last = zeros(N)
    p_last_ref = zeros(N)
    positions = linspace(0.0, (2*pi)-dx, N)
    potential_at_pos = zeros(N)
    drift_at_pos = zeros(N)
    diffusion_at_pos = zeros(N)

    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Launching reference simulation...")
    launchpad_reference(
        positions,
        prob, p_now, p_last, p_last_ref,
        potential_at_pos,
        drift_at_pos, diffusion_at_pos,
        N, dx, check_step, E, psi1, psi2, 
        n, dt, m, beta, gamma
    )
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Reference simulation done!")

    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing data...")

    # recast memoryviews into numpy arrays
    p_now = asarray(p_now)
    p_equil = asarray(prob)
    potential_at_pos = asarray(potential_at_pos)
    drift_at_pos = asarray(drift_at_pos)
    diffusion_at_pos = asarray(diffusion_at_pos)

    # checks to make sure nothing went weird
    # check non-negativity of distribution
    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    # check normalization of distribution
    assert (abs(p_now.sum() - 1.0) <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing finished!")

    # write to file
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data...")
    save_data_reference(
        E, psi1, psi2, n, p_now, p_equil,
        potential_at_pos, drift_at_pos, diffusion_at_pos, N
        )
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!")

    print("Exiting...")

if __name__ == "__main__":
    main()
