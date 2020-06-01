# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False

# ============================================================================
# ==============
# ============== INTERFACES
# ==============
# ============================================================================
def calc_flux(
    double[:] positions, 
    double[:, :] p_now,
    double[:, :, :] drift_at_pos, 
    double[:, :, :] diffusion_at_pos,
    double[:, :, :] flux_array,
    int N, double dx
    ):

    calc_flux_func(
        positions, 
        p_now, 
        drift_at_pos, 
        diffusion_at_pos, 
        flux_array,
        N, dx
        )

def calc_derivative_pxgy(
    double[:, :] p_now, double[:] marginal_now,
    double[:, :] Ly,
    int N, double dx
    ):

    calc_derivative_pxgy_func(p_now, marginal_now, Ly, N, dx)

def step_probability_X(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos,
    int N, double dx, double dt
    ):

    update_probability_x(
        p_now, p_last, 
        drift_at_pos,
        diffusion_at_pos,
        N, dx, dt
        )

# ============================================================================
# ==============
# ============== IMPLEMENTATIONS
# ==============
# ============================================================================
cdef void calc_flux_func(
    double[:] positions, 
    double[:, :] p_now,
    double[:, :, :] drift_at_pos, 
    double[:, :, :] diffusion_at_pos,
    double[:, :, :] flux_array,
    int N, double dx
    ) nogil:

    cdef Py_ssize_t i,j

    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] = (
        (drift_at_pos[0, 0, 0]*p_now[0, 0])
        -(diffusion_at_pos[0, 1, 0]*p_now[1, 0]-diffusion_at_pos[0, N-1, 0]*p_now[N-1, 0])/(2.0*dx)
        -(diffusion_at_pos[1, 0, 1]*p_now[0, 1]-diffusion_at_pos[1, 0, N-1]*p_now[0, N-1])/(2.0*dx)
        )
    flux_array[0, 0, N-1] = (
        (drift_at_pos[0, 0, N-1]*p_now[0, N-1])
        -(diffusion_at_pos[0, 1, N-1]*p_now[1, N-1]-diffusion_at_pos[0, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        -(diffusion_at_pos[1, 0, 0]*p_now[0, 0]-diffusion_at_pos[1, 0, N-2]*p_now[0, N-2])/(2.0*dx)
        )
    flux_array[0, N-1, 0] = (
        (drift_at_pos[0, N-1, 0]*p_now[N-1, 0])
        -(diffusion_at_pos[0, 0, 0]*p_now[0, 0]-diffusion_at_pos[0, N-2, 0]*p_now[N-2, 0])/(2.0*dx)
        -(diffusion_at_pos[1, N-1, 1]*p_now[N-1, 1]-diffusion_at_pos[1, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        )
    flux_array[0, N-1, N-1] = (
        (drift_at_pos[0, N-1, N-1]*p_now[N-1, N-1])
        -(diffusion_at_pos[0, 0, N-1]*p_now[0, N-1]-diffusion_at_pos[0, N-2, N-1]*p_now[N-2, N-1])/(2.0*dx)
        -(diffusion_at_pos[1, N-1, 0]*p_now[N-1, 0]-diffusion_at_pos[1, N-1, N-2]*p_now[N-1, N-2])/(2.0*dx)
        )

    # second component
    flux_array[1, 0, 0] = (
        (drift_at_pos[1, 0, 0]*p_now[0, 0])
        -(diffusion_at_pos[2, 1, 0]*p_now[1, 0]-diffusion_at_pos[2, N-1, 0]*p_now[N-1, 0])/(2.0*dx)
        -(diffusion_at_pos[3, 0, 1]*p_now[0, 1]-diffusion_at_pos[3, 0, N-1]*p_now[0, N-1])/(2.0*dx)
        )
    flux_array[1, 0, N-1] = (
        (drift_at_pos[1, 0, N-1]*p_now[0, N-1])
        -(diffusion_at_pos[2, 1, N-1]*p_now[1, N-1]-diffusion_at_pos[2, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        -(diffusion_at_pos[3, 0, 0]*p_now[0, 0]-diffusion_at_pos[3, 0, N-2]*p_now[0, N-2])/(2.0*dx)
        )
    flux_array[1, N-1, 0] = (
        (drift_at_pos[1, N-1, 0]*p_now[N-1, 0])
        -(diffusion_at_pos[2, 0, 0]*p_now[0, 0]-diffusion_at_pos[2, N-2, 0]*p_now[N-2, 0])/(2.0*dx)
        -(diffusion_at_pos[3, N-1, 1]*p_now[N-1, 1]-diffusion_at_pos[3, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        )
    flux_array[1, N-1, N-1] = (
        (drift_at_pos[1, N-1, N-1]*p_now[N-1, N-1])
        -(diffusion_at_pos[2, 0, N-1]*p_now[0, N-1]-diffusion_at_pos[2, N-2, N-1]*p_now[N-2, N-1])/(2.0*dx)
        -(diffusion_at_pos[3, N-1, 0]*p_now[N-1, 0]-diffusion_at_pos[3, N-1, N-2]*p_now[N-1, N-2])/(2.0*dx)
        )

    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] = (
            (drift_at_pos[0, 0, i]*p_now[0, i])
            -(diffusion_at_pos[0, 1, i]*p_now[1, i]-diffusion_at_pos[0, N-1, i]*p_now[N-1, i])/(2.0*dx)
            -(diffusion_at_pos[1, 0, i+1]*p_now[0, i+1]-diffusion_at_pos[1, 0, i-1]*p_now[0, i-1])/(2.0*dx)
            )
        flux_array[0, i, 0] = (
            (drift_at_pos[0, i, 0]*p_now[i, 0])
            -(diffusion_at_pos[0, i+1, 0]*p_now[i+1, 0]-diffusion_at_pos[0, i-1, 0]*p_now[i-1, 0])/(2.0*dx)
            -(diffusion_at_pos[1, i, 1]*p_now[i, 1]-diffusion_at_pos[1, i, N-1]*p_now[i, N-1])/(2.0*dx)
            )
        flux_array[0, N-1, i] = (
            (drift_at_pos[0, N-1, i]*p_now[N-1, i])
            -(diffusion_at_pos[0, 0, i]*p_now[0, i]-diffusion_at_pos[0, N-2, i]*p_now[N-2, i])/(2.0*dx)
            -(diffusion_at_pos[1, N-1, i+1]*p_now[N-1, i+1]-diffusion_at_pos[1, N-1, i-1]*p_now[N-1, i-1])/(2.0*dx)
            )
        flux_array[0, i, N-1] = (
            (drift_at_pos[0, i, N-1]*p_now[i, N-1])
            -(diffusion_at_pos[0, i+1, N-1]*p_now[i+1, N-1]-diffusion_at_pos[0, i-1, N-1]*p_now[i-1, N-1])/(2.0*dx)
            -(diffusion_at_pos[1, i, 0]*p_now[i, 0]-diffusion_at_pos[1, i, N-2]*p_now[i, N-2])/(2.0*dx)
            )

        # second component
        flux_array[1, 0, i] = (
            (drift_at_pos[1, 0, i]*p_now[0, i])
            -(diffusion_at_pos[2, 1, i]*p_now[1, i]-diffusion_at_pos[2, N-1, i]*p_now[N-1, i])/(2.0*dx)
            -(diffusion_at_pos[3, 0, i+1]*p_now[0, i+1]-diffusion_at_pos[3, 0, i-1]*p_now[0, i-1])/(2.0*dx)
            )
        flux_array[1, i, 0] = (
            (drift_at_pos[1, i, 0]*p_now[i, 0])
            -(diffusion_at_pos[2, i+1, 0]*p_now[i+1, 0]-diffusion_at_pos[2, i-1, 0]*p_now[i-1, 0])/(2.0*dx)
            -(diffusion_at_pos[3, i, 1]*p_now[i, 1]-diffusion_at_pos[3, i, N-1]*p_now[i, N-1])/(2.0*dx)
            )
        flux_array[1, N-1, i] = (
            (drift_at_pos[1, N-1, i]*p_now[N-1, i])
            -(diffusion_at_pos[2, 0, i]*p_now[0, i]-diffusion_at_pos[2, N-2, i]*p_now[N-2, i])/(2.0*dx)
            -(diffusion_at_pos[3, N-1, i+1]*p_now[N-1, i+1]-diffusion_at_pos[3, N-1, i-1]*p_now[N-1, i-1])/(2.0*dx)
            )
        flux_array[1, i, N-1] = (
            (drift_at_pos[1, i, N-1]*p_now[i, N-1])
            -(diffusion_at_pos[2, i+1, N-1]*p_now[i+1, N-1]-diffusion_at_pos[2, i-1, N-1]*p_now[i-1, N-1])/(2.0*dx)
            -(diffusion_at_pos[3, i, 0]*p_now[i, 0]-diffusion_at_pos[3, i, N-2]*p_now[i, N-2])/(2.0*dx)
            )

        # for points with well defined neighbours
        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] = (
                (drift_at_pos[0, i, j]*p_now[i, j])
                -(diffusion_at_pos[0, i+1, j]*p_now[i+1, j]-diffusion_at_pos[0, i-1, j]*p_now[i-1, j])/(2.0*dx)
                -(diffusion_at_pos[1, i, j+1]*p_now[i, j+1]-diffusion_at_pos[1, i, j-1]*p_now[i, j-1])/(2.0*dx)
                )
            # second component
            flux_array[1, i, j] = (
                (drift_at_pos[1, i, j]*p_now[i, j])
                -(diffusion_at_pos[2, i+1, j]*p_now[i+1, j]-diffusion_at_pos[2, i-1, j]*p_now[i-1, j])/(2.0*dx)
                -(diffusion_at_pos[3, i, j+1]*p_now[i, j+1]-diffusion_at_pos[3, i, j-1]*p_now[i, j-1])/(2.0*dx)
                )

cdef void calc_derivative_pxgy_func(
    double[:, :] p_now, double[:] marginal_now,
    double[:, :] out_array, int N, double dx
    ) nogil:

    cdef Py_ssize_t i, j

    # boundary condition on derivative
    out_array[0, 0] = (
        ((p_now[0, 1]/marginal_now[1]) - (p_now[0, N-1]/marginal_now[N-1]))/(2*dx)
        )
    out_array[0, N-1] = (
        ((p_now[0, 0]/marginal_now[0]) - (p_now[0, N-2]/marginal_now[N-2]))/(2*dx)
        )
    out_array[N-1, 0] = (
        ((p_now[N-1, 1]/marginal_now[0]) - (p_now[N-1, N-1]/marginal_now[N-2]))/(2*dx)
        )
    out_array[N-1, N-1] = (
        ((p_now[N-1, 0]/marginal_now[0]) - (p_now[N-1, N-2]/marginal_now[N-2]))/(2*dx)
        )

    # points with well defined neighbors
    for i in range(1, N-1):
        for j in range(1, N-1):
            out_array[i, j] = (
                ((p_now[i, j+1]/marginal_now[j+1]) - (p_now[i, j-1]/marginal_now[j-1]))/(2*dx)
                )

cdef void update_probability_x(
    double[:, :] p_now, 
    double[:, :] p_last, 
    double[:, :, :] drift_at_pos,
    double[:, :, :] diffusion_at_pos, 
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    # Periodic boundary conditions:
    # Explicitly update FPE for the corners
    p_now[0, 0] = p_last[0, 0] + dt*(
        -(drift_at_pos[0, 1, 0]*p_last[1, 0]-drift_at_pos[0, N-1, 0]*p_last[N-1, 0])/(2.0*dx)
        +(diffusion_at_pos[0, 1, 0]*p_last[1, 0]-2.0*diffusion_at_pos[0, 0, 0]*p_last[0, 0]+diffusion_at_pos[0, N-1, 0]*p_last[N-1, 0])/(dx*dx)
        +(diffusion_at_pos[1, 0, 1]*p_last[0, 1]-2.0*diffusion_at_pos[1, 0, 0]*p_last[0, 0]+diffusion_at_pos[1, 0, N-1]*p_last[0, N-1])/(dx*dx)
        ) 
    p_now[0, N-1] = p_last[0, N-1] + dt*(
        -(drift_at_pos[0, 1, N-1]*p_last[1, N-1]-drift_at_pos[0, N-1, N-1]*p_last[N-1, N-1])/(2.0*dx)
        +(diffusion_at_pos[0, 1, N-1]*p_last[1, N-1]-2.0*diffusion_at_pos[0, 0, N-1]*p_last[0, N-1]+diffusion_at_pos[0, N-1, N-1]*p_last[N-1, N-1])/(dx*dx)
        +(diffusion_at_pos[1, 0, 0]*p_last[0, 0]-2.0*diffusion_at_pos[1, 0, N-1]*p_last[0, N-1]+diffusion_at_pos[1, 0, N-2]*p_last[0, N-2])/(dx*dx)
        ) 
    p_now[N-1, 0] = p_last[N-1, 0] + dt*(
        -(drift_at_pos[0, 0, 0]*p_last[0, 0]-drift_at_pos[0, N-2, 0]*p_last[N-2, 0])/(2.0*dx)
        +(diffusion_at_pos[0, 0, 0]*p_last[0, 0]-2.0*diffusion_at_pos[0, N-1, 0]*p_last[N-1, 0]+diffusion_at_pos[0, N-2, 0]*p_last[N-2, 0])/(dx*dx)
        +(diffusion_at_pos[1, N-1, 1]*p_last[N-1, 1]-2.0*diffusion_at_pos[1, N-1, 0]*p_last[N-1, 0]+diffusion_at_pos[1, N-1, N-1]*p_last[N-1, N-1])/(dx*dx)
        ) 
    p_now[N-1, N-1] = p_last[N-1, N-1] + dt*(
        -(drift_at_pos[0, 0, N-1]*p_last[0, N-1]-drift_at_pos[0, N-2, N-1]*p_last[N-2, N-1])/(2.0*dx)
        +(diffusion_at_pos[0, 0, N-1]*p_last[0, N-1]-2.0*diffusion_at_pos[0, N-1, N-1]*p_last[N-1, N-1]+diffusion_at_pos[0, N-2, N-1]*p_last[N-2, N-1])/(dx*dx)
        +(diffusion_at_pos[1, N-1, 0]*p_last[N-1, 0]-2.0*diffusion_at_pos[1, N-1, N-1]*p_last[N-1, N-1]+diffusion_at_pos[1, N-1, N-2]*p_last[N-1, N-2])/(dx*dx)
        ) 

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        # Periodic boundary conditions:
        # Explicitly update FPE for edges not corners
        p_now[0, i] = p_last[0, i] + dt*(
            -(drift_at_pos[0, 1, i]*p_last[1, i]-drift_at_pos[0, N-1, i]*p_last[N-1, i])/(2.0*dx)
            +(diffusion_at_pos[0, 1, i]*p_last[1, i]-2.0*diffusion_at_pos[0, 0, i]*p_last[0, i]+diffusion_at_pos[0, N-1, i]*p_last[N-1, i])/(dx*dx)
            +(diffusion_at_pos[1, 0, i+1]*p_last[0, i+1]-2.0*diffusion_at_pos[1, 1, i]*p_last[0, i]+diffusion_at_pos[1, 0, i-1]*p_last[0, i-1])/(dx*dx)
            ) 
        p_now[i, 0] = p_last[i, 0] + dt*(
            - (drift_at_pos[0, i+1, 0]*p_last[i+1, 0]-drift_at_pos[0, i-1, 0]*p_last[i-1, 0])/(2.0*dx)
            + (diffusion_at_pos[0, i+1, 0]*p_last[i+1, 0]-2.0*diffusion_at_pos[0, i, 0]*p_last[i, 0]+diffusion_at_pos[0, i-1, 0]*p_last[i-1, 0])/(dx*dx)
            + (diffusion_at_pos[1, i, 1]*p_last[i, 1]-2.0*diffusion_at_pos[1, i, 0]*p_last[i, 0]+diffusion_at_pos[1, i, N-1]*p_last[i, N-1])/(dx*dx)
            ) 

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = p_last[i, j] + dt*(
                -(drift_at_pos[0, i+1, j]*p_last[i+1, j]-drift_at_pos[0, i-1, j]*p_last[i-1, j])/(2.0*dx)
                +(diffusion_at_pos[0, i+1, j]*p_last[i+1, j]-2.0*diffusion_at_pos[0, i, j]*p_last[i, j]+diffusion_at_pos[0, i-1, j]*p_last[i-1, j])/(dx*dx)
                +(diffusion_at_pos[1, i, j+1]*p_last[i, j+1]-2.0*diffusion_at_pos[1, i, j]*p_last[i, j]+diffusion_at_pos[1, i, j-1]*p_last[i, j-1])/(dx*dx)
                ) 

        # Explicitly update FPE for rest of edges not corners
        p_now[N-1, i] = p_last[N-1, i] + dt*(
            -(drift_at_pos[0, 0, i]*p_last[0, i]-drift_at_pos[0, N-2, i]*p_last[N-2, i])/(2.0*dx)
            +(diffusion_at_pos[0, 0, i]*p_last[0, i]-2.0*diffusion_at_pos[0, N-1, i]*p_last[N-1, i]+diffusion_at_pos[0, N-2, i]*p_last[N-2, i])/(dx*dx)
            +(diffusion_at_pos[1, N-1, i+1]*p_last[N-1, i+1]-2.0*diffusion_at_pos[1, N-1, i]*p_last[N-1, i]+diffusion_at_pos[1, N-1, i-1]*p_last[N-1, i-1])/(dx*dx)
            ) 
        p_now[i, N-1] = p_last[i, N-1] + dt*(
            -(drift_at_pos[0, i+1, N-1]*p_last[i+1, N-1]-drift_at_pos[0, i-1, N-1]*p_last[i-1, N-1])/(2.0*dx)
            +(diffusion_at_pos[0, i+1, N-1]*p_last[i+1, N-1]-2.0*diffusion_at_pos[0, i, N-1]*p_last[i, N-1]+diffusion_at_pos[0, i-1, N-1]*p_last[i-1, N-1])/(dx*dx)
            +(diffusion_at_pos[1, i, 0]*p_last[i, 0]-2.0*diffusion_at_pos[1, i, N-1]*p_last[i, N-1]+diffusion_at_pos[1, i, N-2]*p_last[i, N-2])/(dx*dx)
            ) 