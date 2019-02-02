from math import pi
from numpy import (
    array, arange, empty, finfo, pi as npi, log, true_divide, asarray,
    empty, zeros, linspace
    )
from time import time
from datetime import datetime
from os.path import isfile

from fpe import (
    launchpad_coupled, launchpad_flows,
    launchpad_nostalgia, launchpad_reference
    )

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
    N = 360  # inverse space discretization. Keep this number high!

    # time-specific parameters
    total_time = 10.0

    # model-specific parameters
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass

    Ax = 3.0 # energy scale of system X
    Axy = 3.0 # energy scale of coupling between systems X and Y
    Ay = 3.0 # energy scale of system Y
    H = 3.0 # force on system X by chemical bath B1
    A = 3.0 # force on system Y by chemical bath B2

    # initialization condition: equilibrium , uniform, or steady
    initialize_from = 'equilibrium'

    # save data or not
    save = True

    # Decide which mode of simulation to run
    # Choices: coupled, nostalgia, flows, save_reference
    mode = "save_reference"

    return (
        ID, mode, dt, N, total_time,
        gamma, beta, m,
        Ax, Ay, Axy, H, A,
        initialize_from, save
        )

def save_data_reference(
    Ax, Axy, Ay, H, A, p_now, p_equil,
    potential_at_pos, force1_at_pos, force2_at_pos,
    N
    ):

    target_dir = './master_output_dir/'
    data_filename = '/reference_Ax_{0}_Axy_{1}_Ay_{2}_Fh_{3}_Fa_{4}_outfile.dat'.format(Ax, Axy, Ay, H, A)
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

def save_data_coupled(
    ID, mode, total_time, Ax, Axy, Ay, H, A,
    p_now, check_sum, p_equil, distance, rel_entropy,
    flux, mean_space_flux, mean_space_time_flux,
    potential_at_pos, force1_at_pos, force2_at_pos,
    N, comp_time
    ):

    target_dir = './master_output_dir/'
    data_filename = '/Coupled_Ax_{0}_Axy_{1}_Ay_{2}_Fh_{3}_Fa_{4}_outfile_{5}.dat'.format(Ax, Axy, Ay, H, A, mode)
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
                "{0}".format(ID) + '\t' #0
                + "{0}".format(Ax) + '\t' #1
                + "{0}".format(Axy) + '\t' #2
                + "{0}".format(Ay) + '\t' #3
                + "{0}".format(H) + '\t' #4
                + "{0}".format(A) + '\t' #5
                + "{0:.15e}".format(comp_time) + '\t' #6
                + "{0:.15e}".format(total_time) + '\t' #7
                + "{0}".format(N) + '\t' #8
                + "{0:.15e}".format(check_sum) + '\t' #9
                + "{0:.15e}".format(distance) + '\t' #10
                + "{0:.15e}".format(rel_entropy) + '\t' #11
                + "{0:.15e}".format(mean_space_flux[0]) + '\t' #12
                + "{0:.15e}".format(mean_space_flux[1]) + '\t' #13
                + "{0:.15e}".format(mean_space_flux.mean()) + '\t' #14
                + "{0:.15e}".format(mean_space_time_flux[0]) + '\t' #15
                + "{0:.15e}".format(mean_space_time_flux[1]) + '\t' # 16
                + "{0:.15e}".format(mean_space_time_flux.mean()) + '\n' #17
                )

def save_data_flows(
    ID, mode, total_time, Ax, Axy, Ay, H, A,
    p_now, check_sum, p_equil, distance, rel_entropy,
    work, heat, nostalgia,
    fluxes_x, fluxes_y,
    mean_space_flux_x, mean_space_time_flux_x,
    mean_space_flux_y, mean_space_time_flux_y,
    potential_at_pos, force1_at_pos, force2_at_pos,
    N, comp_time
    ):

    target_dir = './master_output_dir/'
    matrix_data_filename = '/Flows_Ax_{0}_Axy_{1}_Ay_{2}_Fh_{3}_Fa_{4}_outfile_{5}_matrices.dat'.format(Ax, Axy, Ay, H, A, mode)
    matrix_data_total_path = target_dir + matrix_data_filename

    with open(matrix_data_total_path, 'w') as dfile:
        for i in range(N):
            for j in range(N):
                dfile.write(
                    '{0:.15e}'.format(p_now[i, j])
                    + '\t' + '{0:.15e}'.format(p_equil[i, j])
                    + '\t' + '{0:.15e}'.format(fluxes_x[0, i, j])
                    + '\t' + '{0:.15e}'.format(fluxes_x[1, i, j])
                    + '\t' + '{0:.15e}'.format(fluxes_y[0, i, j])
                    + '\t' + '{0:.15e}'.format(fluxes_y[1, i, j])
                    + '\t' + '{0:.15e}'.format(potential_at_pos[i, j])
                    + '\t' + '{0:.15e}'.format(force1_at_pos[i, j])
                    + '\t' + '{0:.15e}'.format(force2_at_pos[i, j])
                    + '\n'
                )

    summary_filename = '/summary.dat'
    summary_total_path = target_dir + summary_filename

    with open(summary_total_path, 'a') as sfile:

        sfile.write(
                "{0}".format(ID) + '\t' #0
                + "{0}".format(Ax) + '\t' #1
                + "{0}".format(Axy) + '\t' #2
                + "{0}".format(Ay) + '\t' #3
                + "{0}".format(H) + '\t' #4
                + "{0}".format(A) + '\t' #5
                + "{0:.15e}".format(comp_time) + '\t' #6
                + "{0:.15e}".format(total_time) + '\t' #7
                + "{0}".format(N) + '\t' #8
                + "{0:.15e}".format(check_sum) + '\t' #9
                + "{0:.15e}".format(distance) + '\t' #10
                + "{0:.15e}".format(rel_entropy) + '\t' #11
                + "{0:.15e}".format(work) + '\t' #12
                + "{0:.15e}".format(heat) + '\t' #13
                + "{0:.15e}".format(nostalgia) + '\t' #14
                + "{0:.15e}".format(mean_space_flux_x[0]) + '\t' #15
                + "{0:.15e}".format(mean_space_flux_x[1]) + '\t' #16
                + "{0:.15e}".format(mean_space_flux_x.mean()) + '\t' #17
                + "{0:.15e}".format(mean_space_flux_y[0]) + '\t' #18
                + "{0:.15e}".format(mean_space_flux_y[1]) + '\t' #19
                + "{0:.15e}".format(mean_space_flux_y.mean()) + '\t' #20
                + "{0:.15e}".format(mean_space_time_flux_x[0]) + '\t' #21
                + "{0:.15e}".format(mean_space_time_flux_x[1]) + '\t' # 22
                + "{0:.15e}".format(mean_space_time_flux_x.mean()) + '\n' #23
                + "{0:.15e}".format(mean_space_time_flux_y[0]) + '\t' #24
                + "{0:.15e}".format(mean_space_time_flux_y[1]) + '\t' # 25
                + "{0:.15e}".format(mean_space_time_flux_y.mean()) + '\n' #26
                )
def main():

    # unload parameters
    [
        ID, mode, dt, N, total_time, gamma, beta, m, Ax, Ay, Axy, H, A, initialize_from, save
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

    # count number of times the primary loops correspond to the desired
    num_loops = int((total_time+dt)//dt)
    # how many time update steps before checking for steady state convergence
    # enforce steady state convergence check every unit time
    check_step = int(1.0/dt)

    print("Number of times around loop = {0}".format(num_loops))
    print("Number of times before check = {0}".format(check_step))

    if mode.lower() == "save_reference":
        prob = zeros((N, N))
        p_now = zeros((N, N))
        p_last = zeros((N, N))
        p_last_ref = zeros((N, N))
        positions = linspace(0, (2*pi)-dx, N)
        potential_at_pos = zeros((N, N))
        force1_at_pos = zeros((N, N))
        force2_at_pos = zeros((N, N))

        print("{} Launching coupled simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        t0 = time()
        launchpad_reference(
            positions, prob, p_now, p_last, p_last_ref,
            potential_at_pos, force1_at_pos, force2_at_pos,
            N, num_loops,
            dx, check_step,
            Ax, Axy, Ay, H, A,
            dt, m, beta, gamma
        )
        t1 = time()
        comp_time = t1-t0
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

        distance = 0.5*(p_equil - p_now).__abs__().sum(axis=None)
        rel_entropy = p_now.dot(log(p_now / p_equil)).sum(axis=None)

        print("{} Processing finished!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

        # write to file
        print("{} Saving data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        save_data_reference(
            Ax, Axy, Ay, H, A, p_now, p_equil,
            potential_at_pos, force1_at_pos, force2_at_pos, N
            )
        print("{} Saving completed!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

    elif mode.lower() == "coupled":
        prob = zeros((N, N))
        p_now = zeros((N, N))
        p_last = zeros((N, N))
        p_last_ref = zeros((N, N))
        flux = zeros((2, N, N))  # array to keep
        positions = linspace(0, (2*pi)-dx, N)
        potential_at_pos = zeros((N, N))
        force1_at_pos = zeros((N, N))
        force2_at_pos = zeros((N, N))

        print("{} Launching coupled simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        t0 = time()
        launchpad_coupled(
            positions, prob, p_now, p_last, p_last_ref, flux,
            potential_at_pos, force1_at_pos, force2_at_pos,
            N, num_loops,
            dx, time_check, check_step, steady_state_var,
            Ax, Axy, Ay, H, A,
            dt, m, beta, gamma
        )
        t1 = time()
        comp_time = t1-t0
        print("{} Coupled simulation done!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

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
            save_data_reference(
                ID, mode, total_time, Ax, Axy, Ay, H, A,
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

    elif mode.lower() == "nostalgia":
        prob = zeros((N, N))
        p_now = zeros((N, N))
        p_last = zeros((N, N))
        p_last_ref = zeros((N, N))
        p_x_next_y_now = zeros((N, N))
        p_x_next_y_next = zeros((N, N))
        fluxes_x = zeros((2, N, N))  # array to keep
        fluxes_y = zeros((2, N, N))  # array to keep
        positions = linspace(0, (2*pi)-dx, N)
        p_x_now = zeros(N)
        p_y_now = zeros(N)
        p_y_next = zeros(N)
        potential_at_pos = zeros((N, N))
        force1_at_pos = zeros((N, N))
        force2_at_pos = zeros((N, N))

        print("{} Launching nostalgia simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        t0 = time()
        work, heat, nostalgia = launchpad_nostalgia(
            positions, p_x_now, p_y_now, p_y_next,
            prob, p_now, p_last, p_last_ref,
            p_x_next_y_now, p_x_next_y_next,
            fluxes_x, fluxes_y,
            potential_at_pos, force1_at_pos, force2_at_pos,
            N, num_loops,
            dx, time_check, check_step, steady_state_var,
            Ax, Axy, Ay, H, A,
            dt, m, beta, gamma
        )
        t1 = time()
        comp_time = t1-t0
        print("{} Nostalgia simulation done!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

        print("{} Processing data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        # recast everything into a numpy array
        p_now = asarray(p_now) # initial distribution
        p_equil = asarray(prob) # equilibrium distribution absent forcing
        p_last = asarray(p_last) # equilibrium distribution absent forcing
        p_x_next_y_now = asarray(p_x_next_y_now)
        p_x_next_y_next = asarray(p_x_next_y_next)
        fluxes_x = asarray(fluxes_x)
        fluxes_y = asarray(fluxes_y)
        positions = asarray(positions)
        potential_at_pos = asarray(potential_at_pos)
        force1_at_pos = asarray(force1_at_pos)
        force2_at_pos = asarray(force2_at_pos)

        # average over the positions
        mean_space_flux_x = fluxes_x.mean(axis=(1, 2))
        mean_space_time_flux_x = mean_space_flux_x / float(num_loops)
        mean_space_flux_y = fluxes_x.mean(axis=(1, 2))
        mean_space_time_flux_y = mean_space_flux_x / float(num_loops)

        # for checking normalization
        check_sum = p_last.sum(axis=None)

        # checks to make sure nothing went weird
        assert (p_last >= 0.0).all(), \
            "ABORT: Probability density has negative values!"
        assert ((check_sum - 1.0).__abs__() <= finfo('float32').eps), \
            "Check sum = " + str(check_sum) + " ABORT: Probability density is not normalized!"

        distance = 0.5*(p_equil - p_now).__abs__().sum(axis=None)
        rel_entropy = p_now.dot(log(p_now / p_equil)).sum(axis=None)

        print("{} Processing finished!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

        # write to file or to stdout
        if save:
            print("{} Saving data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
            save_data_flows(
                ID, mode, total_time, Ax, Axy, Ay, H, A,
                p_x_next_y_next, check_sum, p_equil, distance, rel_entropy,
                work, heat, nostalgia,
                fluxes_x, fluxes_y,
                mean_space_flux_x, mean_space_time_flux_x,
                mean_space_flux_y, mean_space_time_flux_y,
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
                "Total Variation Distance = {0}, D(P||pi_eq) = {1}, W = {2}, Q = {3}, I_(nos) = {4}".format(distance, rel_entropy, work, heat, nostalgia)
                )
            print("<J1x> = {0}, <J2x> = {1}, <Jx> = {2}".format(mean_space_flux_x[0], mean_space_flux_x[1], mean_space_flux_x.mean()))
            print("<J1y> = {0}, <J2y> = {1}, <Jy> = {2}".format(mean_space_flux_y[0], mean_space_flux_y[1], mean_space_flux_y.mean()))
            print("<<J1x>> = {0}, <<J2x>> = {1}, <<Jx>>= {2}".format(mean_space_time_flux_x[0], mean_space_time_flux_x[1], mean_space_time_flux_x.mean()))
            print("<<J1y>> = {0}, <<J2y>> = {1}, <<Jy>>= {2}".format(mean_space_time_flux_y[0], mean_space_time_flux_y[1], mean_space_time_flux_y.mean()))
            print("="*40)

    elif mode.lower() == "flows":
        work = 0.0
        heat = 0.0
        nostalgia = 0.0
        prob = zeros((N, N))
        p_now = zeros((N, N))
        p_last = zeros((N, N))
        p_last_ref = zeros((N, N))
        p_x_tHalf_y_now = zeros((N, N))
        p_x_tHalf_y_next = zeros((N, N))
        p_x_next_y_now = zeros((N, N))
        p_x_next_y_next = zeros((N, N))
        fluxes_x = zeros((2, N, N))  # array to keep
        fluxes_y = zeros((2, N, N))  # array to keep
        positions = linspace(0, (2*pi)-dx, N)
        p_x_now = zeros(N)
        p_y_now = zeros(N)
        p_y_next = zeros(N)
        potential_at_pos = zeros((N, N))
        force1_at_pos = zeros((N, N))
        force2_at_pos = zeros((N, N))

        print("{} Launching flows simulation...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        t0 = time()
        launchpad_flows(
            positions, p_x_now, p_y_now, p_y_next,
            prob, p_now, p_last, p_last_ref,
            p_x_tHalf_y_now, p_x_tHalf_y_next, p_x_next_y_now, p_x_next_y_next,
            fluxes_x, fluxes_y,
            potential_at_pos, force1_at_pos, force2_at_pos,
            N, num_loops,
            dx, time_check, check_step, steady_state_var,
            Ax, Axy, Ay, H, A,
            dt, m, beta, gamma
        )
        t1 = time()
        comp_time = t1-t0
        print("{} Flows simulation done!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

        print("{} Processing data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
        # recast everything into a numpy array
        p_now = asarray(p_now) # initial distribution
        p_equil = asarray(prob) # equilibrium distribution absent forcing
        p_last = asarray(p_last) # equilibrium distribution absent forcing
        p_x_tHalf_y_now = asarray(p_x_tHalf_y_now)
        p_x_tHalf_y_next = asarray(p_x_tHalf_y_next)
        p_x_next_y_now = asarray(p_x_next_y_now)
        p_x_next_y_next = asarray(p_x_next_y_next)
        fluxes_x = asarray(fluxes_x)
        fluxes_y = asarray(fluxes_y)
        positions = asarray(positions)
        potential_at_pos = asarray(potential_at_pos)
        force1_at_pos = asarray(force1_at_pos)
        force2_at_pos = asarray(force2_at_pos)

        # average over the positions
        mean_space_flux_x = fluxes_x.mean(axis=(1, 2))
        mean_space_time_flux_x = mean_space_flux_x / float(num_loops)
        mean_space_flux_y = fluxes_x.mean(axis=(1, 2))
        mean_space_time_flux_y = mean_space_flux_x / float(num_loops)

        # for checking normalization
        check_sum = p_last.sum(axis=None)

        # checks to make sure nothing went weird
        assert (p_last >= 0.0).all(), \
            "ABORT: Probability density has negative values!"
        assert ((check_sum - 1.0).__abs__() <= finfo('float32').eps), \
            "Check sum = " + str(check_sum) + " ABORT: Probability density is not normalized!"

        distance = 0.5*(p_equil - p_now).__abs__().sum(axis=None)
        rel_entropy = p_now.dot(log(p_now / p_equil)).sum(axis=None)

        print("{} Processing finished!".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))

        # write to file or to stdout
        if save:
            print("{} Saving data...".format(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
            save_data_flows(
                ID, mode, total_time, Ax, Axy, Ay, H, A,
                p_x_next_y_next, check_sum, p_equil, distance, rel_entropy,
                work, heat, nostalgia,
                fluxes_x, fluxes_y,
                mean_space_flux_x, mean_space_time_flux_x,
                mean_space_flux_y, mean_space_time_flux_y,
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
                "Total Variation Distance = {0}, D(P||pi_eq) = {1}, W = {2}, Q = {3}, I_(nos) = {4}".format(distance, rel_entropy, work, heat, nostalgia)
                )
            print("<J1x> = {0}, <J2x> = {1}, <Jx> = {2}".format(mean_space_flux_x[0], mean_space_flux_x[1], mean_space_flux_x.mean()))
            print("<J1y> = {0}, <J2y> = {1}, <Jy> = {2}".format(mean_space_flux_y[0], mean_space_flux_y[1], mean_space_flux_y.mean()))
            print("<<J1x>> = {0}, <<J2x>> = {1}, <<Jx>>= {2}".format(mean_space_time_flux_x[0], mean_space_time_flux_x[1], mean_space_time_flux_x.mean()))
            print("<<J1y>> = {0}, <<J2y>> = {1}, <<Jy>>= {2}".format(mean_space_time_flux_y[0], mean_space_time_flux_y[1], mean_space_time_flux_y.mean()))
            print("="*40)

    print("Exiting...")

if __name__ == "__main__":
    main()
