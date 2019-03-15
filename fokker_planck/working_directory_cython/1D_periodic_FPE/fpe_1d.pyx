# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, log, sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

def launchpad_reference(
    double[:] positions,
    double[:] prob,
    double[:] p_now, double[:] p_last,
    double[:] p_last_ref,
    double[:] potential_at_pos,
    double[:] force_at_pos,
    int N, double dx, unsigned int check_step,
    double A, double H, double atp, double overall, 
    double dt, double m, double beta, double gamma
    ):

    cdef:
        double     Z = 0.0

        Py_ssize_t i # declare iterator variables

    # populate the reference arrays
    for i in range(N):
        potential_at_pos[i] = potential(positions[i], A, H)
        force_at_pos[i] = force(positions[i], A, H, atp, overall)

    # calculate the partition function
    for i in range(N):
        Z += exp(-beta*potential_at_pos[i])

    # calculate the boltzmann equilibrium function and the average energy
    for i in range(N):
        prob[i] = exp(-beta*potential_at_pos[i])/Z

    # initialize the simulation to steady state distribution
    for i in range(N):
        p_now[i] = 1.0/N

    steady_state_initialize(
        positions, p_now, p_last, p_last_ref,
        force_at_pos,
        N, dx, dt, check_step,
        m, gamma, beta
    )

###############################################################################
###############################################################################
############
############ ENERGETICS AND FORCES
############
###############################################################################
###############################################################################

cdef double force(
    double position,
    double A, double H, double atp, double overall
    ) nogil: # force on system X
    return (0.5)*(3*A*sin(3*position)) - overall

cdef double potential(
    double position, double A, double H
    ) nogil:
    return 0.5*(A*(1-cos((3*position))))

###############################################################################
###############################################################################
############
############ INITIALIZATIONS
############
###############################################################################
###############################################################################

cdef void steady_state_initialize(
    double[:] positions,
    double[:] p_now,
    double[:] p_last,
    double[:] p_last_ref,
    double[:] force_at_pos,
    int N, double dx, double dt, unsigned int check_step,
    double m, double gamma, double beta
    ) nogil:

    cdef:
        double              tot_var_dist       = 0.0
        int                 continue_condition = 1

        # counters
        Py_ssize_t          i, j
        unsigned long       step_counter       = 0

    while continue_condition:

        for i in range(N):
            # save previous distribution
            p_last[i] = p_now[i]
            # reset to zero
            p_now[i] = 0.0

        # advance probability one time step
        update_probability_full(
            positions, p_now, p_last,
            force_at_pos,
            m, gamma, beta,
            N, dx, dt
            )

        if step_counter == check_step:
            for i in range(N):
                tot_var_dist += 0.5*fabs(p_last_ref[i] - p_now[i])

            # check condition
            if tot_var_dist < float64_eps:
                continue_condition = 0
            else:
                tot_var_dist = 0.0 # reset total variation distance
                step_counter = 0 # reset step counter
                # make current distribution the reference distribution
                for i in range(N):
                    p_last_ref[i] = p_now[i]

        step_counter += 1

cdef void update_probability_full(
    double[:] positions,
    double[:] p_now,
    double[:] p_last,
    double[:] force_at_pos,
    double m, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i

    ## Periodic boundary conditions:
    ## Explicity update FPE for the ends
    p_now[0] = (
        p_last[0]
        + dt*(force_at_pos[1]*p_last[1]-force_at_pos[N-1]*p_last[N-1])/(gamma*m*2.0*dx)
        + dt*(p_last[1]-2.0*p_last[0]+p_last[N-1])/(beta*gamma*m*(dx*dx))
        ) # checked
    p_now[N-1] = (
        p_last[N-1]
        + dt*(force_at_pos[0]*p_last[0]-force_at_pos[N-2]*p_last[N-2])/(gamma*m*2.0*dx)
        + dt*(p_last[0]-2.0*p_last[N-1]+p_last[N-2])/(beta*gamma*m*(dx*dx))
        ) # checked

    # all points with well defined neighbours go like so:
    for i in range(1, N-1):
        p_now[i] = (
            p_last[i]
            + dt*(force_at_pos[i+1]*p_last[i+1]-force_at_pos[i-1]*p_last[i-1])/(gamma*m*2.0*dx)
            + dt*(p_last[i+1]-2.0*p_last[i]+p_last[i-1])/(beta*gamma*m*(dx*dx))
            ) # checked
