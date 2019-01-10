# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, log, sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
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
    double[:] p_x_now,
    double[:] p_y_now,
    double[:] p_y_next,
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
            p_x_next_y_next[i, j] = p_now[i, j]
            p_last[i, j] = p_now[i, j]

    work, heat, nostalgia = run_simulation_flows(
        positions, p_x_now, p_y_now, p_y_next,
        p_last, p_x_next_y_now, p_x_next_y_next, fluxes_x, fluxes_y,
        potential_at_pos, force1_at_pos, force2_at_pos,
        N, dx, dt, num_loops, m, gamma, beta, E_after_relax
        )

    return work, heat, nostalgia

###############################################################################
###############################################################################
############
############ ENERGETICS AND FORCES
############
###############################################################################
###############################################################################

cdef double force1(
    double position1, double position2,
    double Ax, double Axy, double H
    ) nogil: # force on system X
    return (0.5)*(Axy*sin(position1-position2)+(3*Ax*sin((3*position1)-(2*pi/3)))) + H

cdef double force2(
    double position1, double position2,
    double Ay, double Axy, double A
    ) nogil: # force on system Y
    return (0.5)*((-1.0)*Axy*sin(position1-position2)+(3*Ay*sin(3*position2))) - A

cdef double potential(
    double position1, double position2,
    double Ax, double Axy, double Ay
    ) nogil:
    return 0.5*(Ax*(1-cos((3*position1)-(2*pi/3)))+Axy*(1-cos(position1-position2))+Ay*(1-cos((3*position2))))

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
############ UPDATE STEPS - FULL UPDATES
############
###############################################################################
###############################################################################

cdef void update_probability_x(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] force1_at_pos,
    double m1, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0] = (
            p_last[0, 0]
            + dt*(force1_at_pos[1, 0]*p_last[1, 0]-force1_at_pos[N-1, 0]*p_last[N-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
            ) # checked
    p_now[0, N-1] = (
        p_last[0, N-1]
        + dt*(force1_at_pos[1, N-1]*p_last[1, N-1]-force1_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
        ) # checked
    p_now[N-1, 0] = (
        p_last[N-1, 0]
        + dt*(force1_at_pos[0, 0]*p_last[0, 0]-force1_at_pos[N-2, 0]*p_last[N-2, 0])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
        ) # checked
    p_now[N-1, N-1] = (
        p_last[N-1, N-1]
        + dt*(force1_at_pos[0, N-1]*p_last[0, N-1]-force1_at_pos[N-2, N-1]*p_last[N-2, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i] = (
            p_last[0, i]
            + dt*(force1_at_pos[1, i]*p_last[1, i]-force1_at_pos[N-1, i]*p_last[N-1, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
            ) # checked
        p_now[i, 0] = (
            p_last[i, 0]
            + dt*(force1_at_pos[i+1, 0]*p_last[i+1, 0]-force1_at_pos[i-1, 0]*p_last[i-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = (
                p_last[i, j]
                + dt*(force1_at_pos[i+1, j]*p_last[i+1, j]-force1_at_pos[i-1, j]*p_last[i-1, j])/(gamma*m1*2.0*dx)
                + dt*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i] = (
            p_last[N-1, i]
            + dt*(force1_at_pos[0, i]*p_last[0, i]-force1_at_pos[N-2, i]*p_last[N-2, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
            ) # checked
        p_now[i, N-1] = (
            p_last[i, N-1]
            + dt*(force1_at_pos[i+1, N-1]*p_last[i+1, N-1]-force1_at_pos[i-1, N-1]*p_last[i-1, N-1])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
            ) # checked

cdef void update_probability_y(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] force2_at_pos,
    double m2, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0] = (
            p_last[0, 0]
            + dt*(force2_at_pos[0, 1]*p_last[0, 1]-force2_at_pos[0, N-1]*p_last[0, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
    p_now[0, N-1] = (
        p_last[0, N-1]
        + dt*(force2_at_pos[0, 0]*p_last[0, 0]-force2_at_pos[0, N-2]*p_last[0, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, 0] = (
        p_last[N-1, 0]
        + dt*(force2_at_pos[N-1, 1]*p_last[N-1, 1]-force2_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, N-1] = (
        p_last[N-1, N-1]
        + dt*(force2_at_pos[N-1, 0]*p_last[N-1, 0]-force2_at_pos[N-1, N-2]*p_last[N-1, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i] = (
            p_last[0, i]
            + dt*(force2_at_pos[0, i+1]*p_last[0, i+1]-force2_at_pos[0, i-1]*p_last[0, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, 0] = (
            p_last[i, 0]
            + dt*(force2_at_pos[i, 1]*p_last[i, 1]-force2_at_pos[i, N-1]*p_last[i, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = (
                p_last[i, j]
                + dt*(force2_at_pos[i, j+1]*p_last[i, j+1]-force2_at_pos[i, j-1]*p_last[i, j-1])/(gamma*m2*2.0*dx)
                + dt*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i] = (
            p_last[N-1, i]
            + dt*(force2_at_pos[N-1, i+1]*p_last[N-1, i+1]-force2_at_pos[N-1, i-1]*p_last[N-1, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, N-1] = (
            p_last[i, N-1]
            + dt*(force2_at_pos[i, 0]*p_last[i, 0]-force2_at_pos[i, N-2]*p_last[i, N-2])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked

cdef void update_probability_full(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    double m1, double m2, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0] = (
            p_last[0, 0]
            + dt*(force1_at_pos[1, 0]*p_last[1, 0]-force1_at_pos[N-1, 0]*p_last[N-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[0, 1]*p_last[0, 1]-force2_at_pos[0, N-1]*p_last[0, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
    p_now[0, N-1] = (
        p_last[0, N-1]
        + dt*(force1_at_pos[1, N-1]*p_last[1, N-1]-force1_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
        + dt*(force2_at_pos[0, 0]*p_last[0, 0]-force2_at_pos[0, N-2]*p_last[0, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, 0] = (
        p_last[N-1, 0]
        + dt*(force1_at_pos[0, 0]*p_last[0, 0]-force1_at_pos[N-2, 0]*p_last[N-2, 0])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
        + dt*(force2_at_pos[N-1, 1]*p_last[N-1, 1]-force2_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, N-1] = (
        p_last[N-1, N-1]
        + dt*(force1_at_pos[0, N-1]*p_last[0, N-1]-force1_at_pos[N-2, N-1]*p_last[N-2, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
        + dt*(force2_at_pos[N-1, 0]*p_last[N-1, 0]-force2_at_pos[N-1, N-2]*p_last[N-1, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i] = (
            p_last[0, i]
            + dt*(force1_at_pos[1, i]*p_last[1, i]-force1_at_pos[N-1, i]*p_last[N-1, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[0, i+1]*p_last[0, i+1]-force2_at_pos[0, i-1]*p_last[0, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, 0] = (
            p_last[i, 0]
            + dt*(force1_at_pos[i+1, 0]*p_last[i+1, 0]-force1_at_pos[i-1, 0]*p_last[i-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[i, 1]*p_last[i, 1]-force2_at_pos[i, N-1]*p_last[i, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = (
                p_last[i, j]
                + dt*(force1_at_pos[i+1, j]*p_last[i+1, j]-force1_at_pos[i-1, j]*p_last[i-1, j])/(gamma*m1*2.0*dx)
                + dt*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                + dt*(force2_at_pos[i, j+1]*p_last[i, j+1]-force2_at_pos[i, j-1]*p_last[i, j-1])/(gamma*m2*2.0*dx)
                + dt*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i] = (
            p_last[N-1, i]
            + dt*(force1_at_pos[0, i]*p_last[0, i]-force1_at_pos[N-2, i]*p_last[N-2, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[N-1, i+1]*p_last[N-1, i+1]-force2_at_pos[N-1, i-1]*p_last[N-1, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, N-1] = (
            p_last[i, N-1]
            + dt*(force1_at_pos[i+1, N-1]*p_last[i+1, N-1]-force1_at_pos[i-1, N-1]*p_last[i-1, N-1])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[i, 0]*p_last[i, 0]-force2_at_pos[i, N-2]*p_last[i, N-2])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked

###############################################################################
###############################################################################
############
############ DYNAMICAL QUANTITIES
############
###############################################################################
###############################################################################

cdef void calc_flux(
    double[:] positions,
    double[:, :]  p_now,
    double[:, :, :]  flux_array,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    double m1, double m2, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    cdef Py_ssize_t i, j # declare iterator variables

    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] += (-1.0)*(
        (force1_at_pos[0, 0]*p_now[0, 0])/(gamma*m1)
        + (p_now[1, 0] - p_now[N-1, 0])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
    flux_array[0, 0, N-1] += (-1.0)*(
        (force1_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m1)
        + (p_now[1, N-1] - p_now[N-1, N-1])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, 0] += (-1.0)*(
        (force1_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m1)
        + (p_now[0, 0] - p_now[N-2, 0])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, N-1] += (-1.0)*(
        (force1_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m1)
        + (p_now[0, N-1] - p_now[N-2, N-1])/(beta*gamma*m1*2*dx)
        )*(dt/dx)

    # second component
    flux_array[1, 0, 0] += (-1.0)*(
        (force2_at_pos[0, 0]*p_now[0, 0])/(gamma*m2)
        + (p_now[0, 1] - p_now[0, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, 0, N-1] += (-1.0)*(
        (force2_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m2)
        + (p_now[0, 0] - p_now[0, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, 0] += (-1.0)*(
        (force2_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m2)
        + (p_now[N-1, 1] - p_now[N-1, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, N-1] += (-1.0)*(
        (force2_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m2)
        + (p_now[N-1, 0] - p_now[N-1, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)

    # for points with well defined neighbours
    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] += (-1.0)*(
            (force1_at_pos[0, i]*p_now[0, i])/(gamma*m1)
            + (p_now[1, i] - p_now[N-1, i])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
        flux_array[0, i, 0] += (-1.0)*(
            (force1_at_pos[i, 0]*p_now[i, 0])/(gamma*m1)
            + (p_now[i+1, 0]- p_now[i-1, 0])/(beta*gamma*m1*2*dx)
        )*(dt/dx)

        # second component
        flux_array[1, 0, i] += (-1.0)*(
            (force2_at_pos[0, i]*p_now[0, i])/(gamma*m2)
            + (p_now[0, i+1] - p_now[0, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, 0] += (-1.0)*(
            (force2_at_pos[i, 0]*p_now[i, 0])/(gamma*m2)
            + (p_now[i, 1] - p_now[i, N-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)

        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] += (-1.0)*(
                (force1_at_pos[i, j]*p_now[i, j])/(gamma*m1)
                + (p_now[i+1, j] - p_now[i-1, j])/(beta*gamma*m1*2*dx)
                )*(dt/dx)
            # second component
            flux_array[1, i, j] += (-1.0)*(
                (force2_at_pos[i, j]*p_now[i, j])/(gamma*m2)
                + (p_now[i, j+1] - p_now[i, j-1])/(beta*gamma*m2*2*dx)
                )*(dt/dx)

        # update rest of edges not corners
        # first component
        flux_array[0, N-1, i] += (-1.0)*(
            (force1_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m1)
            + (p_now[0, i] - p_now[N-2, i])/(beta*gamma*m1*2*dx)
            )*(dt/dx)
        flux_array[0, i, N-1] += (-1.0)*(
            (force1_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m1)
            + (p_now[i+1, N-1] - p_now[i-1, N-1])/(beta*gamma*m1*2*dx)
            )*(dt/dx)

        # second component
        flux_array[1, N-1, i] += (-1.0)*(
            (force2_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m2)
            + (p_now[N-1, i+1] - p_now[N-1, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, N-1] += (-1.0)*(
            (force2_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m2)
            + (p_now[i, 0] - p_now[i, N-2])/(beta*gamma*m2*2*dx)
            )*(dt/dx)

###############################################################################
###############################################################################
############
############ INFORMATION THEORETIC QUANTITIES
############
###############################################################################
###############################################################################
cdef double calc_transfer_entropy(None) nogil:
    # first compute H(Y_t|Y_{t-1:t-L}), unsure how to compute
    # next compute H(Y_t|Y_{t-1:t-L}, X_{t-1:t-L}), unsure how to compute
    # compute the transfer entropy by subtraction
    return 0.0

cdef double calc_learning_rate(None) nogil:
    # first compute I(X_{t};Y_{t+\tau})
    # then compute derivative wrt \tau
    return 0.0

cdef double calc_nostalgia(
    double[:] p_x_now, double[:] p_x_next, double[:] p_y_now,
    double[:, :] p_xy_now, double[:, :] p_x_next_y_now,
    int N, double dx
    ) nogil:

    cdef:
        double I_mem  = 0.0
        double I_pred = 0.0

        # declare iterator variables
        Py_ssize_t i, j

    # compute the marginal distributions for the "now" variables
    for i in range(N):
        for j in range(N):
            p_x_now[i] += p_xy_now[i, j]
    for j in range(N):
        for i in range(N):
            p_y_now[j] += p_xy_now[i, j]

    # compute I_{mem} = I(X_{t}, Y_{t}), assume dx = dy
    for i in range(N):
        for j in range(N):
            I_mem += p_xy_now[i, j]*log(p_xy_now[i, j]/(p_x_now[i]*p_y_now[j]))
    I_mem *= (dx*dx)

    # reset the marginal distributions
    for i in range(N):
        p_x_now[i] = 0.0
        p_y_now[i] = 0.0
        p_y_next[i] = 0.0

    # compute the marginal distributions for the "next" variables
    for i in range(N):
        for j in range(N):
            p_x_next[i] += p_x_next_y_now[i, j]
    for j in range(N):
        for i in range(N):
            p_y_now[j] += p_x_next_y_now[i, j]

    # compute I_{pred} = I(X_{t}, Y_{t+1})
    for i in range(N):
        for j in range(N):
            I_pred += p_x_next_y_now[i, j]*log(p_x_next_y_now[i, j]/(p_x_next[i]*p_y_now[j]))
    I_pred *= (dx*dx)

    # compute the nostalgia by subtraction
    return I_mem - I_pred

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

cdef (double, double, double) run_simulation_flows(
    double[:] positions,
    double[:] p_x_now,
    double[:] p_y_now,
    double[:] p_y_next,
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
        double      nostalgia          = 0.0
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

        nostalgia += calc_nostalgia(
            p_x_now, p_x_next, p_y_now,
            p_last, p_x_next_y_now,
            N, dx
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

    return work, heat, nostalgia
