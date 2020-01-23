#!/usr/bin/env python3

from math import pi
from numpy import finfo, zeros, set_printoptions, ones, asarray, empty
from datetime import datetime

from initialize import problem_2D

from os import mkdir, path

import spectral_mod

from sys import stderr


def get_params():

    # discretization parameters
    dt = 5e-2  # time discretization. Keep this number low
    N = 360  # inverse space discretization. Keep this number high!

    D = 0.001  # diffusion constant of problem; inverse zeta

    E0 = 3.0  # energy scale of subsystem 1
    Ecouple = 3.0  # degree of coupling
    E1 = 3.0  # energy scale of subsystem 2
    psi1 = 3.0  # energy INTO (positive) subsystem 1 by chemical bath 1
    psi2 = 3.0  # energy INTO (positive) subsystem 2 by chemical bath 2

    n1 = 3.0  # number of minima in the potential of system 1
    n2 = 3.0  # number of minima in the potential of system 2
    phase = 0.0  # how much subsystems are offset from one another

    # define the problem
    problem = problem_2D(
        n=N, m=N, E0=E0, Ec=Ecouple, E1=E1,
        num_minima0=n1, num_minima1=n2, phase=phase
        D=D, psi0=psi1, psi1=psi2
    )

    return (dt, problem)


def save_data_reference(
    p_now, drift_at_pos, diffusion_at_pos, problem
):

    # unack the problem
    n1 = problem.num_minima0
    n2 = problem.num_minima1

    E0 = problem.E0
    Ecouple = problem.Ec
    E1 = problem.E1

    psi1 = problem.psi0
    psi2 = problem.psi1

    potential_at_pos = problem.Epot

    p_equil = problem.p_equil

    # target_dir = '../../../master_output_dir/'
    target_dir = '/Users/jlucero/data_dir/2020-01-08/'

    if not path.isdir(target_dir):
        print("Target directory doesn't exist. Making it now.")
        mkdir(target_dir)

    data_filename = (
        f"/reference_E0_{E0}_Ecouple_{Ecouple}_E1_{E1}_"
        + f"psi1_{psi1}_psi2_{psi2}_"
        + f"n1_{n1}_n2_{n2}_phase_0.0_"
        + "outfile.dat"
    )
    data_total_path = target_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(problem.n):
            for j in range(problem.m):
                dfile.write(
                    f'{p_now[i, j]:.15e}\t'  # 0
                    + f'{p_equil[i, j]:.15e}\t'  # 1
                    + f'{potential_at_pos[i, j]:.15e}\t'  # 2
                    + f'{drift_at_pos[0, i, j]:.15e}\t'  # 3
                    + f'{drift_at_pos[1, i, j]:.15e}\t'  # 4
                    + f'{diffusion_at_pos[0, i, j]:.15e}\t'  # 5
                    + f'{diffusion_at_pos[1, i, j]:.15e}\t'  # 6
                    + f'{diffusion_at_pos[2, i, j]:.15e}\t'  # 7
                    + f'{diffusion_at_pos[3, i, j]:.15e}\n'  # 8
                )


def main():

    # unload parameters
    [dt, problem] = get_params()

    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print(f"Number of times before check = {check_step}")

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Prepping reference simulation..."
    )

    # set initial distribution to be the uniform distribution
    # p_initial = asarray(problem.p_equil, order="F")
    p_initial = ones((problem.n, problem.m))/(problem.n*problem.m)

    # initialize array which holds the steady state distribution
    p_ss = zeros((problem.n, problem.m), order="F")

    drift1 = asarray(problem.mu1, order="F")
    drift2 = asarray(problem.mu2, order="F")
    ddrift1 = asarray(problem.dmu1, order="F")
    ddrift2 = asarray(problem.dmu2, order="F")

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Starting FPE integration..."
    )

    spectral_mod.fft_solve.get_spectral_steady(
        dt, check_step, problem.D, problem.dx, problem.dy,
        drift1, ddrift1, drift2, ddrift2,
        p_initial, p_ss, problem.n, problem.m
    )
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "FPE integration done!"
    )

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing data...")

    # set all small enough numbers to zero
    p_ss[p_ss.__abs__() <= finfo("float64").eps] = 0.0

    assert (p_ss >= 0.0).all(), \
        "ABORT: Probability density has negative values!"

    assert (abs(p_ss.sum(axis=None) - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is no longer normalized!"

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Processing finished!"
    )

    # write to file
    print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data...")

    # construct drift tensors
    drift_tensor = empty((2, problem.n, problem.m))
    drift_tensor[0, ...] = problem.D*drift1[...]
    drift_tensor[1, ...] = problem.D*drift2[...]

    # construct diffusion tensor
    diffusion_tensor = zeros((4, problem.n, problem.m))
    diffusion_tensor[0, ...] = problem.D
    diffusion_tensor[3, ...] = problem.D

    # discard other no longer needed arrays
    del drift1, drift2, ddrift1, ddrift2,

    save_data_reference(
        p_ss, drift_tensor, diffusion_tensor, problem
    )

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!")

    print("Exiting...")


if __name__ == "__main__":
    main()
