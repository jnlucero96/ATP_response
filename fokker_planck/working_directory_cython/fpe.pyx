# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, log, sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
# float32 machine eps
cdef double float32_eps = 1.1920928955078125e-07
# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

def launchpad_reference(
    double num_minima1, double num_minima2, 
    double phase_shift,
    double[:] positions,
    double[:, :] prob, double[:, :] p_now,
    double[:, :] p_last, double[:, :] p_last_ref,
    double[:, :] potential_at_pos,
    double[:, :] force1_at_pos, double[:, :] force2_at_pos,
    double[:, :] rotation_check,
    int N, double dx, unsigned int check_step,
    double E0, double Ecouple, double E1, double F_Hplus, double F_atp,
    double dt, double m, double beta, double gamma, int rotation_index
    ):

    cdef:
        double     Z = 0.0

        Py_ssize_t i, j # declare iterator variables

    # populate the reference arrays
    for i in range(N):
        for j in range(N):
            potential_at_pos[i, j] = potential(
                positions[i], positions[j], 
                num_minima1, num_minima2, phase_shift, E0, Ecouple, E1
                )
            force1_at_pos[i, j] = force1(
                positions[i], positions[j], 
                num_minima1, num_minima2, phase_shift, E0, Ecouple, F_Hplus
                )
            force2_at_pos[i, j] = force2(
                positions[i], positions[j], 
                num_minima1, num_minima2, E1, Ecouple, F_atp
                )

    # calculate the partition function
    for i in range(N):
        for j in range(N):
            Z += exp(-beta*potential_at_pos[i, j])

    # calculate the boltzmann equilibrium function and the average energy
    for i in range(N):
        for j in range(N):
            prob[i, j] = exp(-beta*potential_at_pos[i, j])/Z

    # initialize the simulation to steady state distribution
    for i in range(N):
        for j in range(N):
            p_now[i, j] = 1.0/(N*N)

    steady_state_initialize(
        p_now, p_last, p_last_ref,
        force1_at_pos, force2_at_pos,
        rotation_check,
        N, dx, dt, check_step,
        m, gamma, beta, rotation_index
    )

###############################################################################
###############################################################################
############
############ ENERGETICS AND FORCES
############
###############################################################################
###############################################################################

cdef double force1(
    double position1, double position2, 
    double num_minima1, double num_minima2, 
    double phase_shift,
    double E0, double Ecouple, double F_Hplus
    ) nogil:
    # Returns the force on system F0. H+ chemical potential set up so that
    # postive values of chemical potential returns postive values of the flux
    # for F0
    return (0.5)*(
        Ecouple*sin(position1-position2)
        + (num_minima1*E0*sin((num_minima1*position1)-(phase_shift)))
        ) - F_Hplus

cdef double force2(
    double position1, double position2,
    double num_minima1, double num_minima2,
    double E1, double Ecouple, double F_atp
    ) nogil:
    # Returns the force on system F1. Chemical potential set up so that
    # postive values of chemical potential returns positive values of the of
    # the flux for F1
    return (0.5)*(
        (-1.0)*Ecouple*sin(position1-position2)
        + (num_minima2*E1*sin(num_minima2*position2))
        ) - F_atp

cdef double potential(
    double position1, double position2, 
    double num_minima1, double num_minima2,
    double phase_shift, double E0, double Ecouple, double E1
    ) nogil:
    return 0.5*(
        E0*(1-cos((num_minima1*position1-phase_shift)))
        + Ecouple*(1-cos(position1-position2))
        + E1*(1-cos((num_minima2*position2)))
        )

###############################################################################
###############################################################################
############
############ INITIALIZATIONS
############
###############################################################################
###############################################################################

cdef void steady_state_initialize(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] p_last_ref,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    double[:, :] rotation_check,
    int N, double dx, double dt, unsigned int check_step,
    double m, double gamma, double beta, int rotation_index
    ) nogil:

    cdef:
        double              m1                 = m
        double              m2                 = m
        double              tot_var_dist       = 0.0
        double              rot_var_dist       = 0.0
        int                 continue_condition = 1

        # counters
        Py_ssize_t          i, j, ii, jj
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
            p_now, p_last,
            force1_at_pos, force2_at_pos,
            m1, m2, gamma, beta,
            N, dx, dt
            )

        if step_counter == check_step:
            for i in range(N):
                for j in range(N):
                    tot_var_dist += 0.5*fabs(p_last_ref[i, j]-p_now[i, j])

            # check convergence criteria #1
            if tot_var_dist < float64_eps:

                rotate_distribution(N, rotation_index, rotation_check, p_now)

                # check convergence criteria #2
                for ii in range(N):
                    for jj in range(N):
                        rot_var_dist += 0.5*fabs(rotation_check[i, j]-p_now[i, j])
                
                if rot_var_dist < float64_eps:
                    continue_condition = 0
                else: 
                    tot_var_dist = 0.0 # reset total variation distance
                    step_counter = 0 # reset step counter
                    for i in range(N):
                        for j in range(N):
                            # make current distribution the reference 
                            # distribution
                            p_last_ref[i, j] = p_now[i, j]
                            # reset rotation_check
                            rotation_check[i, j] = 0.0

            else:
                tot_var_dist = 0.0 # reset total variation distance
                step_counter = 0 # reset step counter
                # make current distribution the reference distribution
                for i in range(N):
                    for j in range(N):
                        p_last_ref[i, j] = p_now[i, j]

        step_counter += 1

cdef void rotate_distribution(
    int N, int rotation_index, double[:, :] rotation_check, 
    double[:, :] p_now
    ) nogil:

    cdef Py_ssize_t i, j

    for i in range(N): 
        for j in range(N):
            rotation_check[i, j] = p_now[(i-rotation_index) % N, (j-rotation_index) % N]

cdef void update_probability_full(
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
            + dt*(p_last[1, i]-2.0*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[0, i+1]*p_last[0, i+1]-force2_at_pos[0, i-1]*p_last[0, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, i+1]-2.0*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, 0] = (
            p_last[i, 0]
            + dt*(force1_at_pos[i+1, 0]*p_last[i+1, 0]-force1_at_pos[i-1, 0]*p_last[i-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, 0]-2.0*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[i, 1]*p_last[i, 1]-force2_at_pos[i, N-1]*p_last[i, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 1]-2.0*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
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