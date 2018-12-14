# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs

# import self-made cython modules
from energetics cimport potential, force1, force2
from updates cimport (
    update_probability_full, update_probability_x, update_probability_y
    )
from dynamical cimport calc_flux
from information_theoretic cimport (
    calc_nostalgia, calc_learning_rate, calc_transfer_entropy
    )

# float32 machine eps
cdef double float32_eps = 1.1920928955078125e-07
# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

def launchpad_coupled(
    double[:] positions,
    double[:, :] prob, double[:, :] p_now, double[:, :] p_last,
    double[:, :] p_last_ref, double[:, :, :] flux,
    double[:, :] potential_at_pos,
    double[:, :] force1_at_pos, double[:, :] force2_at_pos,
    int N, int num_loops,
    double dx, double time_check, unsigned int check_step, int steady_state,
    double Ax, double Axy, double Ay, double H, double A,
    double dt, double m, double beta, double gamma
    ):

    cdef:
        double     Z = 0.0
        double     E = 0.0
        double     work
        double     heat
        double     mean_flux
        double     p_sum
        double     E_after_relax
        double     E_0

        Py_ssize_t i, j # declare iterator variables

    # populate the reference arrays
    for i in range(N):
        for j in range(N):
            potential_at_pos[i, j] = potential(positions[i], positions[j], Ax, Axy, Ay)
            force1_at_pos[i, j] = force1(positions[i], positions[j], Ax, Axy, H)
            force2_at_pos[i, j] = force2(positions[i], positions[j], Ay, Axy, A)

    # calculate the partition function
    for i in range(N):
        for j in range(N):
            Z += exp(-beta*potential_at_pos[i, j])

    # calculate the boltzmann equilibrium function and the average energy
    for i in range(N):
        for j in range(N):
            prob[i, j] = exp(-beta*potential_at_pos[i, j])/Z
            E += potential_at_pos[i, j]*exp(-beta*potential_at_pos[i, j])/Z

    # initialize the simulation from the appropriate distribution
    if steady_state == 0:
        for i in range(N):
            for j in range(N):
                p_now[i, j] = prob[i, j]
    elif steady_state == 1:
        for i in range(N):
            for j in range(N):
                p_now[i, j] = 1.0/(N*N)
    else:
        for i in range(N):
            for j in range(N):
                p_now[i, j] = 1.0/(N*N)
        steady_state_initialize(
            positions, p_now, p_last, p_last_ref,
            force1_at_pos, force2_at_pos,
            N, dx, dt, num_loops, check_step,
            m, gamma, beta
        )

    E_after_relax = E
    E_0 = E

    run_simulation_coupled(
        positions, p_now, p_last, flux,
        potential_at_pos, force1_at_pos, force2_at_pos,
        N, dx, dt, num_loops,
        m, gamma, beta, E_after_relax
    )

def launchpad_flows(
    double[:] positions,
    double[:, :] prob,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] p_last_ref,
    double[:, :] p_x_next_y_now,
    double[:, :] p_x_next_y_next,
    double[:, :, :] fluxes_x,
    double[:, :, :] fluxes_y,
    double[:, :] potential_at_pos,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    int N, int num_loops,
    double dx, double time_check, unsigned int check_step, int steady_state,
    double Ax, double Axy, double Ay, double H, double A,
    double dt, double m, double beta, double gamma
    ):

    cdef:
        double     Z = 0.0
        double     E = 0.0
        double     work
        double     heat
        double     mean_flux
        double     p_sum
        double     E_after_relax
        double     E_0

        Py_ssize_t i, j # declare iterator variables

    # populate the reference arrays
    for i in range(N):
        for j in range(N):
            potential_at_pos[i, j] = potential(positions[i], positions[j], Ax, Axy, Ay)
            force1_at_pos[i, j] = force1(positions[i], positions[j], Ax, Axy, H)
            force2_at_pos[i, j] = force2(positions[i], positions[j], Ay, Axy, A)

    # calculate the partition function
    for i in range(N):
        for j in range(N):
            Z += exp(-beta*potential_at_pos[i, j])

    # calculate the boltzmann equilibrium function and the average energy
    for i in range(N):
        for j in range(N):
            prob[i, j] = exp(-beta*potential_at_pos[i, j])/Z
            E += potential_at_pos[i, j]*exp(-beta*potential_at_pos[i, j])/Z

    # initialize the simulation from the appropriate distribution
    if steady_state == 0:
        for i in range(N):
            for j in range(N):
                p_now[i, j] = prob[i, j]
    elif steady_state == 1:
        for i in range(N):
            for j in range(N):
                p_now[i, j] = 1.0/(N*N)
    else:
        for i in range(N):
            for j in range(N):
                p_now[i, j] = 1.0/(N*N)
        steady_state_initialize(
            positions, p_now, p_last, p_last_ref,
            force1_at_pos, force2_at_pos,
            N, dx, dt, num_loops, check_step,
            m, gamma, beta
        )

    E_after_relax = E
    E_0 = E

    for i in range(N):
        for j in range(N):
            p_last[i, j] = p_now[i, j]
            p_now[i, j] = 0.0

    run_simulation_flows(
        positions,
        p_last,
        p_x_next_y_now,
        p_x_next_y_next,
        fluxes_x,
        fluxes_y,
        potential_at_pos,
        force1_at_pos,
        force2_at_pos,
        N, dx, dt, num_loops, m, gamma, beta, E_after_relax
        )

###############################################################################
###############################################################################
############
############ INITIALIZATIONS
############
###############################################################################
###############################################################################

cdef void steady_state_initialize(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] p_last_ref,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    int N, double dx, double dt, int num_loops, unsigned int check_step,
    double m, double gamma, double beta
    ) nogil:

    cdef:
        double              work               = 0.0  # Cumulative work
        double              heat               = 0.0  # Cumulative heat
        double              E_last             = 0.0
        double              E_change_pot       = 0.0
        double              m1                 = m
        double              m2                 = m
        double              tot_var_dist       = 0.0
        int                 continue_condition = 1

        # counters
        Py_ssize_t          i, j
        unsigned long       step_counter       = 0

    while continue_condition:

        for i in range(N):
            for j in range(N):
            # save previous distribution
                p_last[i, j] = p_now[i, j]
                # reset to zero
                p_now[i, j] = 0.0

        # advance probability one time step
        update_probability_full(
            positions, p_now, p_last,
            force1_at_pos, force2_at_pos,
            m1, m2, gamma, beta,
            N, dx, dt
            )

        if step_counter == check_step:
            for i in range(N):
                for j in range(N):
                    tot_var_dist += 0.5*fabs(p_last_ref[i, j] - p_now[i, j])

            # check condition
            if tot_var_dist < float64_eps:
                continue_condition = 0
            else:
                tot_var_dist = 0.0 # reset total variation distance
                step_counter = 0 # reset step counter
                # make current distribution the reference distribution
                for i in range(N):
                    for j in range(N):
                        p_last_ref[i, j] = p_now[i, j]

        step_counter += 1

###############################################################################
###############################################################################
############
############ SIMULATIONS
############
###############################################################################
###############################################################################

cdef void run_simulation_coupled(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :, :] flux,
    double[:, :] potential_at_pos,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    int N, double dx, double dt, int num_loops,
    double m, double gamma, double beta, double E_after_relax
    ) nogil:

    cdef:
        double      m1                 = m
        double      m2                 = m

        # counters
        Py_ssize_t         i, j, n

    for n in range(num_loops):

        for i in range(N):
            for j in range(N):
            # save previous distribution
                p_last[i, j] = p_now[i, j]
                # reset to zero
                p_now[i, j] = 0.0

        update_probability_full(
            positions, p_now, p_last,
            force1_at_pos, force2_at_pos,
            m1, m2, gamma, beta,
            N, dx, dt
            )

        calc_flux(
            positions, p_now, flux,
            force1_at_pos, force2_at_pos,
            m1, m2, gamma, beta,
            N, dx, dt
            )

cdef (double, double) run_simulation_flows(
    double[:] positions,
    double[:, :] p_last,
    double[:, :] p_x_next_y_now,
    double[:, :] p_x_next_y_next,
    double[:, :, :] fluxes_x,
    double[:, :, :] fluxes_y,
    double[:, :] potential_at_pos,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    int N, double dx, double dt, int num_loops,
    double m, double gamma, double beta, double E_after_relax
    ) nogil:

    cdef:
        double      work               = 0.0  # Cumulative work
        double      heat               = 0.0  # Cumulative heat
        double      t                  = 0.0  # time
        double      E_last             = 0.0
        double      E_change_pot       = 0.0
        double      m1                 = m
        double      m2                 = m
        int         continue_condition = 1

        # counters
        Py_ssize_t         i, j, n

    for n in range(num_loops):

        for i in range(N):
            for j in range(N):
            # save previous distribution
                p_last[i, j] = p_x_next_y_next[i, j]
                # reset to zero
                p_x_next_y_now[i, j] = 0.0
                p_x_next_y_next[i, j] = 0.0

        update_probability_x(
            positions, p_x_next_y_now, p_last, force1_at_pos,
            m1, gamma, beta, N, dx, dt
            )

        E_last = E_after_relax  #Energy at t-dt is E_last
        for i in range(N):
            for j in range(N):
                E_change_pot += potential_at_pos[i, j]*p_x_next_y_now[i, j]
        work += E_change_pot - E_last #add to cumulative work

        calc_flux(
            positions, p_x_next_y_now, fluxes_x,
            force1_at_pos, force2_at_pos,
            m1, m2, gamma, beta,
            N, dx, dt
            )

        update_probability_y(
            positions, p_x_next_y_next, p_x_next_y_now, force2_at_pos,
            m2, gamma, beta, N, dx, dt
            )

        for i in range(N):
            for j in range(N):
                E_after_relax += potential_at_pos[i,j]*p_x_next_y_next[i, j]
        heat += E_after_relax - E_change_pot # adds to cumulative heat

        calc_flux(
            positions, p_x_next_y_next, fluxes_y,
            force1_at_pos, force2_at_pos,
            m1, m2, gamma, beta,
            N, dx, dt
            )

        # reset energy variables
        E_change_pot = 0.0
        E_after_relax = 0.0

    return work, heat