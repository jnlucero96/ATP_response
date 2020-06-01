# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

def launchpad_reference(
    double[:] positions,
    double[:] prob, double[:] p_now, double[:] p_last, double[:] p_last_ref,
    double[:] potential_at_pos,
    double[:] drift_at_pos, double[:] diffusion_at_pos,
    int N, double dx, unsigned int check_step,
    double E, double psi1, double psi2,
    double n, double dt, double m, double beta, double gamma
    ):

    cdef:
        double     Z = 0.0

        Py_ssize_t i # declare iterator variables

    # populate the reference arrays
    for i in range(N):
        potential_at_pos[i] = potential(positions[i], E, n)
        drift_at_pos[i] = drift(positions[i], E, psi1, psi2, n, m, beta, gamma)
        diffusion_at_pos[i] = diffusion(positions[i], m, beta, gamma)

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
        p_now, p_last, p_last_ref,
        drift_at_pos, diffusion_at_pos,
        N, dx, dt, check_step,
        m, gamma, beta
    )

# =============================================================================
# =============================================================================
# ==========
# ========== ENERGETICS AND FORCES
# ==========
# =============================================================================
# =============================================================================

# position dependent drift on system 
cdef double drift(
    double position, double E, double psi1, double psi2, double n,
    double m, double beta, double gamma
    ) nogil: 
    #return -((1.0/(m*gamma))*(0.5*(n*E*sin(n*position))-(psi1+psi2))) #F1 ATPase
    return -0.5*((1.0/(m*gamma))*(0.5*(n*E*sin(n*position))-(psi1+psi2))) #infinitely strong coupled FoF1

# position dependent diffusion on system 
cdef double diffusion(
    double position, double m, double beta, double gamma
    ) nogil: 
    #return 1.0/(beta*m*gamma) #F1 ATPase
    return 1.0/(2*beta*m*gamma) #infinitely strong coupled FoF1
    
# potential of system
cdef double potential(double position, double E, double n) nogil: 
    #return 0.5*E*(1-cos((n*position)))
    return 0.25*E*(1-cos((n*position))) #infinitely strong coupled FoF1

# =============================================================================
# =============================================================================
# ==========
# ========== INITIALIZATIONS
# ==========
# =============================================================================
# =============================================================================

cdef void steady_state_initialize(
    double[:] p_now,
    double[:] p_last,
    double[:] p_last_ref,
    double[:] drift_at_pos,
    double[:] diffusion_at_pos,
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
            p_now, p_last,
            drift_at_pos, diffusion_at_pos,
            m, gamma, beta,
            N, dx, dt
            )

        if step_counter == check_step:
            for i in range(N):
                tot_var_dist += 0.5*fabs(p_last_ref[i]-p_now[i])

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
    double[:] p_now,
    double[:] p_last,
    double[:] drift_at_pos,
    double[:] diffusion_at_pos,
    double m, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i

    # Periodic boundary conditions:
    # Explicity update FPE for the ends
    p_now[0] = p_last[0] + dt*(
        -(drift_at_pos[1]*p_last[1]-drift_at_pos[N-1]*p_last[N-1])/(2.0*dx)
        +(diffusion_at_pos[1]*p_last[1]-2.0*diffusion_at_pos[0]*p_last[0]+diffusion_at_pos[N-1]*p_last[N-1])/(dx*dx)
        ) 
    p_now[N-1] = p_last[N-1] + dt*(
        -(drift_at_pos[0]*p_last[0]-drift_at_pos[N-2]*p_last[N-2])/(2.0*dx)
        +(diffusion_at_pos[0]*p_last[0]-2.0*diffusion_at_pos[N-1]*p_last[N-1]+diffusion_at_pos[N-2]*p_last[N-2])/(dx*dx)
        ) 

    # all points with well defined neighbours go like so:
    for i in range(1, N-1):
        p_now[i] = p_last[i] + dt*(
            -(drift_at_pos[i+1]*p_last[i+1]-drift_at_pos[i-1]*p_last[i-1])/(2.0*dx)
            +(diffusion_at_pos[i+1]*p_last[i+1]-2.0*diffusion_at_pos[i]*p_last[i]+diffusion_at_pos[i-1]*p_last[i-1])/(dx*dx)
            ) 
