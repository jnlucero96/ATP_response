# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, sin, cos
from cython.parallel import prange

# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

# =============================================================================
# =============================================================================
# ==========
# ========== INTERFACES
# ==========
# =============================================================================
# =============================================================================

def launchpad_reference(
    double n1, double n2,
    double phase,
    double[:] positions,
    double[:, :] prob, double[:, :] p_now,
    double[:, :] p_last, double[:, :] p_last_ref,
    double[:, :] potential_at_pos,
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos,
    int N, double dx, unsigned int check_step,
    double E0, double Ecouple, double E1, double mu_Hp, double mu_atp,
    double dt, double m1, double m2, double beta, double gamma1, double gamma2
    ):

    cdef:
        double     Z = 0.0

        Py_ssize_t i, j # declare iterator variables

    # populate the reference arrays
    for i in range(N):
        for j in range(N):
            potential_at_pos[i, j] = potential(
                positions[i], positions[j],
                n1, n2, phase, E0, Ecouple, E1
                )

    # calculate the partition function
    for i in range(N):
        for j in range(N):
            Z += exp(-beta*potential_at_pos[i, j])

    # calculate the boltzmann equilibrium function and the average energy
    for i in range(N):
        for j in range(N):
            prob[i, j] = exp(-beta*potential_at_pos[i, j])/Z

    # build the drift vector
    construct_drift_vec_func(
        drift_at_pos, positions,
        E0, Ecouple, E1, mu_Hp, mu_atp,
        phase, n1, n2, m1, m2,
        gamma1, gamma2, N
        )

    # build diffusion tensor
    construct_diffusion_tensor_func(
        diffusion_at_pos, positions, m1, m2, beta, gamma1, gamma2, N
        )

    # initialize the simulation to steady state distribution
    # start with Gibbs-Boltzmann distribution as initial guess
    for i in range(N):
        for j in range(N):
            p_now[i, j] = prob[i, j]

    steady_state_initialize(
        p_now, p_last, p_last_ref,
        drift_at_pos, diffusion_at_pos,
        N, dx, dt, check_step
    )

def construct_drift_vec(
    double[:, :, :] drift_at_pos, double[:] positions,
    double E0, double Ecouple, double E1, double mu_Hp, double mu_atp,
    double phase, double n1, double n2, double m1, double m2,
    double gamma1, double gamma2, int N
    ):

    construct_drift_vec_func(
        drift_at_pos, positions,
        E0, Ecouple, E1, mu_Hp, mu_atp,
        phase, n1, n2, m1, m2,
        gamma1, gamma2, N
        )

def construct_diffusion_tensor(
    double[:, :, :] diffusion_at_pos, double[:] positions,
    double m1, double m2, double beta, double gamma1, double gamma2, int N
    ):

    construct_diffusion_tensor_func(
        diffusion_at_pos, positions, m1, m2, beta, gamma1, gamma2, N
        )

# =============================================================================
# =============================================================================
# ==========
# ========== CONSTRUCTOR FUNCTIONS
# ==========
# =============================================================================
# =============================================================================

cdef void construct_drift_vec_func(
    double[:, :, :] drift_at_pos, double[:] positions,
    double E0, double Ecouple, double E1, double mu_Hp, double mu_atp,
    double phase, double n1, double n2, double m1, double m2,
    double gamma1, double gamma2, int N
    ) nogil:

    cdef Py_ssize_t i, j

    for i in range(N):
        for j in range(N):
            drift_at_pos[0, i, j] = drift1(
                positions[i], positions[j],
                n1, phase, E0, Ecouple, mu_Hp,
                m1, gamma1
                )
            drift_at_pos[1, i, j] = drift2(
                positions[i], positions[j],
                n2, E1, Ecouple, mu_atp,
                m2, gamma2
                )

cdef void construct_diffusion_tensor_func(
    double[:, :, :] diffusion_at_pos, double[:] positions,
    double m1, double m2, double beta, double gamma1, double gamma2,
    int N
    ) nogil:

    cdef Py_ssize_t i, j

    for i in range(N):
        for j in range(N):
            diffusion_at_pos[0, i, j] = diffusion11(
                positions[i], positions[j],
                m1, beta, gamma1
                )
            diffusion_at_pos[1, i, j] = diffusion12(
                positions[i], positions[j],
                m1, m2, beta, gamma1, gamma2
                )
            diffusion_at_pos[2, i, j] = diffusion21(
                positions[i], positions[j],
                m1, m2, beta, gamma1, gamma2
                )
            diffusion_at_pos[3, i, j] = diffusion22(
                positions[i], positions[j],
                m2, beta, gamma2
                )

# =============================================================================
# =============================================================================
# ==========
# ========== ENERGETICS AND FORCES
# ==========
# =============================================================================
# =============================================================================

cdef double potential(
    double position1, double position2,
    double n1, double n2,
    double phase, double E0, double Ecouple, double E1
    ) nogil:
    return 0.5*(
        E0*(1-cos(n1*(position1-phase)))
        + Ecouple*(1-cos(position1-position2))
        + E1*(1-cos((n2*position2)))
        )

cdef double drift1(
    double position1, double position2, double n1,
    double phase, double E0, double Ecouple, double mu_Hp,
    double m1, double gamma1
    ) nogil:
    # Returns the force on system F0. H+ chemical potential set up so that
    # postive values of chemical potential returns postive values of the flux
    # for F0
    return (-1.0/(m1*gamma1))*((0.5)*(
        Ecouple*sin(position1-position2)
        + (n1*E0*sin(n1*(position1-phase)))
        ) - mu_Hp)

cdef double drift2(
    double position1, double position2, double n2,
    double E1, double Ecouple, double mu_atp,
    double m2, double gamma2
    ) nogil:
    # Returns the force on system F1. Chemical potential \mu_{ATP} set up so
    # that postive values of chemical potential returns positive values of
    # the of the flux for F1
    return (-1.0/(m2*gamma2))*((0.5)*(
        (-1.0)*Ecouple*sin(position1-position2)
        + (n2*E1*sin(n2*position2))
        ) - mu_atp)

cdef double diffusion11(
    double position1, double position2,
    double m1, double beta, double gamma1
    ) nogil:
    # return the 11 element of the diffusion tensor
    return 1.0/(beta*m1*gamma1)

cdef double diffusion12(
    double position1, double position2,
    double m1, double m2, double beta, double gamma1, double gamma2
    ) nogil:
    # return the 12 element of the diffusion tensor
    return 0.0

cdef double diffusion21(
    double position1, double position2,
    double m1, double m2, double beta, double gamma1, double gamma2
    ) nogil:
    # return the 21 element of the diffusion tensor
    return 0.0

cdef double diffusion22(
    double position1, double position2,
    double m2, double beta, double gamma2
    ) nogil:
    # return the 22 element of the diffusion tensor
    return 1.0/(beta*m2*gamma2)

# =============================================================================
# =============================================================================
# ==========
# ========== INITIALIZATIONS
# ==========
# =============================================================================
# =============================================================================

cdef void steady_state_initialize(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] p_last_ref,
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos,
    int N, double dx, double dt, unsigned int check_step,
    ) nogil:

    cdef:
        double              tot_var_dist       = 0.0
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
            p_now,
            p_last,
            drift_at_pos,
            diffusion_at_pos,
            N, dx, dt
            )

        if step_counter == check_step:
            for i in range(N):
                for j in range(N):
                    tot_var_dist += 0.5*fabs(p_last_ref[i, j]-p_now[i, j])

            # check convergence criteria
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

cdef void update_probability_full(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos,
    int N, double dx, double dt
    ) nogil:

    # update the distribution using FTCS method.

    # declare iterator variables
    cdef Py_ssize_t i, j

    # Periodic boundary conditions:
    # Explicit update FPE for the corners
    p_now[0, 0] = p_last[0, 0] + dt*(
        -(drift_at_pos[0, 1, 0]*p_last[1, 0]-drift_at_pos[0, N-1, 0]*p_last[N-1, 0])/(2.0*dx)
        +(diffusion_at_pos[0, 1, 0]*p_last[1, 0]-2.0*diffusion_at_pos[0, 0, 0]*p_last[0, 0]+diffusion_at_pos[0, N-1, 0]*p_last[N-1, 0])/(dx*dx)
        +(diffusion_at_pos[1, 1, 1]*p_last[1, 1]-diffusion_at_pos[1, 1, N-1]*p_last[1, N-1]-diffusion_at_pos[1, N-1, 1]*p_last[N-1, 1]+diffusion_at_pos[1, N-1, N-1]*p_last[N-1, N-1])/(4.0*dx*dx)
        -(drift_at_pos[1, 0, 1]*p_last[0, 1]-drift_at_pos[1, 0, N-1]*p_last[0, N-1])/(2.0*dx)
        +(diffusion_at_pos[2, 1, 1]*p_last[1, 1]-diffusion_at_pos[2, N-1, 1]*p_last[N-1, 1]-diffusion_at_pos[2, 1, N-1]*p_last[1, N-1]+diffusion_at_pos[2, N-1, N-1]*p_last[N-1, N-1])/(4.0*dx*dx)
        +(diffusion_at_pos[3, 0, 1]*p_last[0, 1]-2.0*diffusion_at_pos[3, 0, 0]*p_last[0, 0]+diffusion_at_pos[3, 0, N-1]*p_last[0, N-1])/(dx*dx)
        )
    p_now[0, N-1] = p_last[0, N-1] + dt*(
        -(drift_at_pos[0, 1, N-1]*p_last[1, N-1]-drift_at_pos[0, N-1, N-1]*p_last[N-1, N-1])/(2.0*dx)
        +(diffusion_at_pos[0, 1, N-1]*p_last[1, N-1]-2.0*diffusion_at_pos[0, 0, N-1]*p_last[0, N-1]+diffusion_at_pos[0, N-1, N-1]*p_last[N-1, N-1])/(dx*dx)
        +(diffusion_at_pos[1, 1, 0]*p_last[1, 0]-diffusion_at_pos[1, 1, N-2]*p_last[1, N-2]-diffusion_at_pos[1, N-1, 0]*p_last[N-1, 0]+diffusion_at_pos[1, N-1, N-2]*p_last[N-1, N-2])/(4.0*dx*dx)
        -(drift_at_pos[1, 0, 0]*p_last[0, 0]-drift_at_pos[1, 0, N-2]*p_last[0, N-2])/(2.0*dx)
        +(diffusion_at_pos[2, 1, 0]*p_last[1, 0]-diffusion_at_pos[2, N-1, 0]*p_last[N-1, 0]-diffusion_at_pos[2, 1, N-2]*p_last[1, N-2]+diffusion_at_pos[2, N-1, N-2]*p_last[N-1, N-2])/(4.0*dx*dx)
        +(diffusion_at_pos[3, 0, 0]*p_last[0, 0]-2.0*diffusion_at_pos[3, 0, N-1]*p_last[0, N-1]+diffusion_at_pos[3, 0, N-2]*p_last[0, N-2])/(dx*dx)
        )
    p_now[N-1, 0] = p_last[N-1, 0] + dt*(
        -(drift_at_pos[0, 0, 0]*p_last[0, 0]-drift_at_pos[0, N-2, 0]*p_last[N-2, 0])/(2.0*dx)
        +(diffusion_at_pos[0, 0, 0]*p_last[0, 0]-2.0*diffusion_at_pos[0, N-1, 0]*p_last[N-1, 0]+diffusion_at_pos[0, N-2, 0]*p_last[N-2, 0])/(dx*dx)
        +(diffusion_at_pos[1, 0, 1]*p_last[0, 1]-diffusion_at_pos[1, 0, N-1]*p_last[0, N-1]-diffusion_at_pos[1, N-2, 1]*p_last[N-2, 1]+diffusion_at_pos[1, N-2, N-1]*p_last[N-2, N-1])/(4.0*dx*dx)
        -(drift_at_pos[1, N-1, 1]*p_last[N-1, 1]-drift_at_pos[1, N-1, N-1]*p_last[N-1, N-1])/(2.0*dx)
        +(diffusion_at_pos[2, 0, 1]*p_last[0, 1]-diffusion_at_pos[2, N-2, 1]*p_last[N-2, 1]-diffusion_at_pos[2, 0, N-1]*p_last[0, N-1]+diffusion_at_pos[2, N-2, N-1]*p_last[N-2, N-1])/(4.0*dx*dx)
        +(diffusion_at_pos[3, N-1, 1]*p_last[N-1, 1]-2.0*diffusion_at_pos[3, N-1, 0]*p_last[N-1, 0]+diffusion_at_pos[3, N-1, N-1]*p_last[N-1, N-1])/(dx*dx)
        )
    p_now[N-1, N-1] = p_last[N-1, N-1] + dt*(
        -(drift_at_pos[0, 0, N-1]*p_last[0, N-1]-drift_at_pos[0, N-2, N-1]*p_last[N-2, N-1])/(2.0*dx)
        +(diffusion_at_pos[0, 0, N-1]*p_last[0, N-1]-2.0*diffusion_at_pos[0, N-1, N-1]*p_last[N-1, N-1]+diffusion_at_pos[0, N-2, N-1]*p_last[N-2, N-1])/(dx*dx)
        +(diffusion_at_pos[1, 0, 0]*p_last[0, 0]-diffusion_at_pos[1, 0, N-2]*p_last[0, N-2]-diffusion_at_pos[1, N-2, 0]*p_last[N-2, 0]+diffusion_at_pos[1, N-2, N-2]*p_last[N-2, N-2])/(4.0*dx*dx)
        -(drift_at_pos[1, N-1, 0]*p_last[N-1, 0]-drift_at_pos[1, N-1, N-2]*p_last[N-1, N-2])/(2.0*dx)
        +(diffusion_at_pos[2, 0, 0]*p_last[0, 0]-diffusion_at_pos[2, N-2, 0]*p_last[N-2, 0]-diffusion_at_pos[2, 0, N-2]*p_last[0, N-2]+diffusion_at_pos[2, N-2, N-2]*p_last[N-2, N-2])/(4.0*dx*dx)
        +(diffusion_at_pos[3, N-1, 0]*p_last[N-1, 0]-2.0*diffusion_at_pos[3, N-1, N-1]*p_last[N-1, N-1]+diffusion_at_pos[3, N-1, N-2]*p_last[N-1, N-2])/(dx*dx)
        )

    # iterate through all the coordinates (not corners) for both variables
    for i in prange(1, N-1):
        # Periodic boundary conditions:
        # Explicitly update FPE for edges of grid (not corners)
        p_now[0, i] = p_last[0, i] + dt*(
            -(drift_at_pos[0, 1, i]*p_last[1, i]-drift_at_pos[0, N-1, i]*p_last[N-1, i])/(2.0*dx)
            +(diffusion_at_pos[0, 1, i]*p_last[1, i]-2.0*diffusion_at_pos[0, 0, i]*p_last[0, i]+diffusion_at_pos[0, N-1, i]*p_last[N-1, i])/(dx*dx)
            +(diffusion_at_pos[1, 1, i+1]*p_last[1, i+1]-diffusion_at_pos[1, 1, i-1]*p_last[1, i-1]-diffusion_at_pos[1, N-1, i+1]*p_last[N-1, i+1]+diffusion_at_pos[1, N-1, i-1]*p_last[N-1, i-1])/(4.0*dx*dx)
            -(drift_at_pos[1, 0, i+1]*p_last[0, i+1]-drift_at_pos[1, 0, i-1]*p_last[0, i-1])/(2.0*dx)
            +(diffusion_at_pos[2, 1, i+1]*p_last[1, i+1]-diffusion_at_pos[2, N-1, i+1]*p_last[N-1, i+1]-diffusion_at_pos[2, 1, i-1]*p_last[1, i-1]+diffusion_at_pos[2, N-1, i-1]*p_last[N-1, i-1])/(4.0*dx*dx)
            +(diffusion_at_pos[3, 0, i+1]*p_last[0, i+1]-2.0*diffusion_at_pos[3, 0, i]*p_last[0, i]+diffusion_at_pos[3, 0, i-1]*p_last[0, i-1])/(dx*dx)
            )
        p_now[i, 0] = p_last[i, 0] + dt*(
            -(drift_at_pos[0, i+1, 0]*p_last[i+1, 0]-drift_at_pos[0, i-1, 0]*p_last[i-1, 0])/(2.0*dx)
            +(diffusion_at_pos[0, i+1, 0]*p_last[i+1, 0]-2.0*diffusion_at_pos[0, i, 0]*p_last[i, 0]+diffusion_at_pos[0, i-1, 0]*p_last[i-1, 0])/(dx*dx)
            +(diffusion_at_pos[1, i+1, 1]*p_last[i+1, 1]-diffusion_at_pos[1, i+1, N-1]*p_last[i+1, N-1]-diffusion_at_pos[1, i-1, 1]*p_last[i-1, 1]+diffusion_at_pos[1, i-1, N-1]*p_last[i-1, N-1])/(4.0*dx*dx)
            -(drift_at_pos[1, i, 1]*p_last[i, 1]-drift_at_pos[1, i, N-1]*p_last[i, N-1])/(2.0*dx)
            +(diffusion_at_pos[2, i+1, 1]*p_last[i+1, 1]-diffusion_at_pos[2, i-1, 1]*p_last[i-1, 1]-diffusion_at_pos[2, i+1, N-1]*p_last[i+1, N-1]+diffusion_at_pos[2, i-1, N-1]*p_last[i-1, N-1])/(4.0*dx*dx)
            +(diffusion_at_pos[3, i, 1]*p_last[i, 1]-2.0*diffusion_at_pos[3, i, 0]*p_last[i, 0]+diffusion_at_pos[3, i, N-1]*p_last[i, N-1])/(dx*dx)
            )
        p_now[N-1, i] = p_last[N-1, i] + dt*(
            -(drift_at_pos[0, 0, i]*p_last[0, i]-drift_at_pos[0, N-2, i]*p_last[N-2, i])/(2.0*dx)
            +(diffusion_at_pos[0, 0, i]*p_last[0, i]-2.0*diffusion_at_pos[0, N-1, i]*p_last[N-1, i]+diffusion_at_pos[0, N-2, i]*p_last[N-2, i])/(dx*dx)
            +(diffusion_at_pos[1, 0, i+1]*p_last[0, i+1]-diffusion_at_pos[1, 0, i-1]*p_last[0, i-1]-diffusion_at_pos[1, N-2, i+1]*p_last[N-2, i+1]+diffusion_at_pos[1, N-2, i-1]*p_last[N-2, i-1])/(4.0*dx*dx)
            -(drift_at_pos[1, N-1, i+1]*p_last[N-1, i+1]-drift_at_pos[1, N-1, i-1]*p_last[N-1, i-1])/(2.0*dx)
            +(diffusion_at_pos[2, 0, i+1]*p_last[0, i+1]-diffusion_at_pos[2, N-2, i+1]*p_last[N-2, i+1]-diffusion_at_pos[2, 0, i-1]*p_last[0, i-1]+diffusion_at_pos[2, N-2, i-1]*p_last[N-2, i-1])/(4.0*dx*dx)
            +(diffusion_at_pos[3, N-1, i+1]*p_last[N-1, i+1]-2.0*diffusion_at_pos[3, N-1, i]*p_last[N-1, i]+diffusion_at_pos[3, N-1, i-1]*p_last[N-1, i-1])/(dx*dx)
            )
        p_now[i, N-1] = p_last[i, N-1] + dt*(
            -(drift_at_pos[0, i+1, N-1]*p_last[i+1, N-1]-drift_at_pos[0, i-1, N-1]*p_last[i-1, N-1])/(2.0*dx)
            +(diffusion_at_pos[0, i+1, N-1]*p_last[i+1, N-1]-2.0*diffusion_at_pos[0, i, N-1]*p_last[i, N-1]+diffusion_at_pos[0, i-1, N-1]*p_last[i-1, N-1])/(dx*dx)
            +(diffusion_at_pos[1, i+1, 0]*p_last[i+1, 0]-diffusion_at_pos[1, i+1, N-2]*p_last[i+1, N-2]-diffusion_at_pos[1, i-1, 0]*p_last[i-1, 0]+diffusion_at_pos[1, i-1, N-2]*p_last[i-1, N-2])/(4.0*dx*dx)
            -(drift_at_pos[1, i, 0]*p_last[i, 0]-drift_at_pos[1, i, N-2]*p_last[i, N-2])/(2.0*dx)
            +(diffusion_at_pos[2, i+1, 0]*p_last[i+1, 0]-diffusion_at_pos[2, i-1, 0]*p_last[i-1, 0]-diffusion_at_pos[2, i+1, N-2]*p_last[i+1, N-2]+diffusion_at_pos[2, i-1, N-2]*p_last[i-1, N-2])/(4.0*dx*dx)
            +(diffusion_at_pos[3, i, 0]*p_last[i, 0]-2.0*diffusion_at_pos[3, i, N-1]*p_last[i, N-1]+diffusion_at_pos[3, i, N-2]*p_last[i, N-2])/(dx*dx)
            )

        # all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = p_last[i, j] + dt*(
                -(drift_at_pos[0, i+1, j]*p_last[i+1, j]-drift_at_pos[0, i-1, j]*p_last[i-1, j])/(2.0*dx)
                +(diffusion_at_pos[0, i+1, j]*p_last[i+1, j]-2.0*diffusion_at_pos[0, i, j]*p_last[i, j]+diffusion_at_pos[0, i-1, j]*p_last[i-1, j])/(dx*dx)
                +(diffusion_at_pos[1, i+1, j+1]*p_last[i+1, j+1]-diffusion_at_pos[1, i+1, j-1]*p_last[i+1, j-1]-diffusion_at_pos[1, i-1, j+1]*p_last[i-1, j+1]+diffusion_at_pos[1, i-1, j-1]*p_last[i-1, j-1])/(4.0*dx*dx)
                -(drift_at_pos[1, i, j+1]*p_last[i, j+1]-drift_at_pos[1, i, j-1]*p_last[i, j-1])/(2.0*dx)
                +(diffusion_at_pos[2, i+1, j+1]*p_last[i+1, j+1]-diffusion_at_pos[2, i-1, j+1]*p_last[i-1, j+1]-diffusion_at_pos[2, i+1, j-1]*p_last[i+1, j-1]+diffusion_at_pos[2, i-1, j-1]*p_last[i-1, j-1])/(4.0*dx*dx)
                +(diffusion_at_pos[3, i, j+1]*p_last[i, j+1]-2.0*diffusion_at_pos[3, i, j]*p_last[i, j]+diffusion_at_pos[3, i, j-1]*p_last[i, j-1])/(dx*dx)
                )
