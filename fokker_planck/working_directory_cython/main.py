#!/usr/bin/env python3
from math import pi
from numpy import finfo, asarray, empty, zeros, linspace
from datetime import datetime

from fpe import launchpad_reference


def get_params():

    # discretization parameters
    dt = 0.001  # time discretization. Keep this number low
    N = 540  # inverse space discretization. Keep this number high!

    # model-specific parameters
    gamma1 = 1000.0  # drag coefficient of subsystem 1
    gamma2 = 1000.0  # drag coefficient of subsystem 2
    beta = 1.0  # thermodynamic beta: 1/kT
    m1 = 1.0  # mass of subsystem 1
    m2 = 1.0  # mass of subsystem 2

    E0 = 2.0 # energy scale of subsystem 1
    Ecouple = 1.0 # energy scale of coupling between subsystems 1 and 2
    E1 = 2.0 # energy scale of subsystem 2
    psi1 = 4.0 #  energy INTO (positive) subsystem 1 by chemical bath 1
    psi2 = 2.0 # energy INTO (positive) subsystem 2 by chemical bath 2

    n1 = 3.0  # number of minima in the potential of system 1
    n2 = 3.0  # number of minima in the potential of system 2
    phase = 0.0  # how much sub-systems are offset from one another

    return (
        dt, N, gamma1, gamma2, beta, m1, m2, n1, n2,
        phase, E0, E1, Ecouple, psi1, psi2
    )


def save_data_reference(
    n1, n2,
    phase,
    E0, Ecouple, E1, psi1, psi2, p_now, p_equil,
    potential_at_pos, drift_at_pos, diffusion_at_pos,
    N
):

    target_dir = '../../../../master_output_dir/'
    data_filename = (
        f"/reference_E0_{E0}_Ecouple_{Ecouple}_E1_{E1}_"
        + f"psi1_{psi1}_psi2_{psi2}_"
        + f"n1_{n1}_n2_{n2}_phase_{phase}_"
        + "outfile.dat"
    )
    data_total_path = target_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            for j in range(N):
                dfile.write(
                    f'{p_now[i, j]:.15e}'  # 0
                    + '\t' + f'{p_equil[i, j]:.15e}'  # 1
                    + '\t' + f'{potential_at_pos[i, j]:.15e}'  # 2
                    + '\t' + f'{drift_at_pos[0, i, j]:.15e}'  # 3
                    + '\t' + f'{drift_at_pos[1, i, j]:.15e}'  # 4
                    + '\t' + f'{diffusion_at_pos[0, i, j]:.15e}'  # 5
                    + '\t' + f'{diffusion_at_pos[1, i, j]:.15e}'  # 6
                    + '\t' + f'{diffusion_at_pos[2, i, j]:.15e}'  # 7
                    + '\t' + f'{diffusion_at_pos[3, i, j]:.15e}'  # 8
                    + '\n'
                )


def main():

    # unload parameters
    [
        dt, N, gamma1, gamma2, beta, m1, m2, n1, n2,
        phase, E0, E1, Ecouple, psi1, psi2
    ] = get_params()

    # calculate derived discretization parameters
    dx = (2*pi) / N  # space discretization: total distance / number of points

    # provide CSL criteria to make sure simulation doesn't blow up
    if E0 == 0.0 and E1 == 0.0:
        time_check = 100000000.0
    else:
        time_check = dx/(
            (0.5*(Ecouple+E0*n1)-psi1)/(m1*gamma1)
            + (0.5*(Ecouple+E1*n2)-psi2)/(m2*gamma2)
        )

    if dt > time_check:
        # bail if user is stupid
        print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
        exit(1)

    # how many time update steps before checking for steady state convergence
    # enforce steady state convergence check every unit time
    check_step = int(2.0/dt)

    print(f"Number of times before check = {check_step}")

    prob = zeros((N, N))
    p_now = zeros((N, N))
    p_last = zeros((N, N))
    p_last_ref = zeros((N, N))
    positions = linspace(0, (2*pi)-dx, N)
    potential_at_pos = zeros((N, N))
    drift_at_pos = zeros((2, N, N))
    diffusion_at_pos = zeros((4, N, N))

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Launching FPE simulation..."
    )
    launchpad_reference(
        n1, n2,
        phase,
        positions,
        prob, p_now,
        p_last, p_last_ref,
        potential_at_pos,
        drift_at_pos,
        diffusion_at_pos,
        N, dx, check_step,
        E0, Ecouple, E1, psi1, psi2,
        dt, m1, m2, beta, gamma1, gamma2
    )
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "FPE simulation done!"
    )

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Processing data..."
    )
    # recast everything into a numpy array
    p_now = asarray(p_now)
    p_equil = asarray(prob)
    potential_at_pos = asarray(potential_at_pos)
    drift_at_pos = asarray(drift_at_pos)
    diffusion_at_pos = asarray(diffusion_at_pos)

    # checks to make sure nothing went weird: bail at first sign of trouble
    # check the non-negativity of the distribution
    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    # check the normalization
    assert (abs(p_now.sum(axis=None) - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Processing finished!"
    )

    # write to file
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data..."
    )
    save_data_reference(
        n1, n2,
        phase,
        E0, Ecouple, E1, psi1, psi2, p_now, p_equil,
        potential_at_pos, drift_at_pos, diffusion_at_pos, N
    )
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!")
    print("Exiting...")


if __name__ == "__main__":
    main()
