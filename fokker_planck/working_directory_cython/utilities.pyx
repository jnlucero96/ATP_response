# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, log, sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
# float32 machine eps
cdef double float32_eps = 1.1920928955078125e-07
# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

# ============================================================================
# ==============
# ============== INTERFACES
# ==============
# ============================================================================
def calc_flux(
    double[:] positions, double[:, :] p_now,
    double[:, :] force1_at_pos, double[:, :] force2_at_pos,
    double[:, :, :] flux_array,
    double m1, double m2, double gamma, double beta, int N, double dx
    ):

    calc_flux_func(
        positions, p_now, force1_at_pos, force2_at_pos, flux_array,
        m1, m2, gamma, beta, N, dx
        )

def calc_learning_rate(
    double[:, :] p_now, double[:] marginal_now,
    double[:, :] flux_array,
    double[:, :] Ly,
    int N, double dx
    ):

    calc_learning_rate_func(p_now, marginal_now, flux_array, Ly, N, dx)

def calc_derivative_pxgy(
    double[:, :] p_now, double[:] marginal_now,
    double[:, :] Ly,
    int N, double dx
    ):

    calc_derivative_pxgy_func(p_now, marginal_now, Ly, N, dx)

def step_probability_X(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] force1_at_pos,
    double m1, double gamma, double beta,
    int N, double dx, double dt
    ):

    update_probability_x(
        p_now, p_last, force1_at_pos,
        m1, gamma, beta, N, dx, dt
        )

# ============================================================================
# ==============
# ============== IMPLEMENTATIONS
# ==============
# ============================================================================
cdef void calc_flux_func(
    double[:] positions, double[:, :] p_now,
    double[:, :] force1_at_pos, double[:, :] force2_at_pos,
    double[:, :, :] flux_array,
    double m1, double m2, double gamma, double beta, int N, double dx
    ) nogil:

    cdef Py_ssize_t i,j

    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] = (-1.0)*(
        (force1_at_pos[0, 0]*p_now[0, 0])/(gamma*m1)
        + (p_now[1, 0] - p_now[N-1, 0])/(beta*gamma*m1*2*dx)
        )
    flux_array[0, 0, N-1] = (-1.0)*(
        (force1_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m1)
        + (p_now[1, N-1] - p_now[N-1, N-1])/(beta*gamma*m1*2*dx)
        )
    flux_array[0, N-1, 0] = (-1.0)*(
        (force1_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m1)
        + (p_now[0, 0] - p_now[N-2, 0])/(beta*gamma*m1*2*dx)
        )
    flux_array[0, N-1, N-1] = (-1.0)*(
        (force1_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m1)
        + (p_now[0, N-1] - p_now[N-2, N-1])/(beta*gamma*m1*2*dx)
        )

    # second component
    flux_array[1, 0, 0] = (-1.0)*(
        (force2_at_pos[0, 0]*p_now[0, 0])/(gamma*m2)
        + (p_now[0, 1] - p_now[0, N-1])/(beta*gamma*m2*2*dx)
        )
    flux_array[1, 0, N-1] = (-1.0)*(
        (force2_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m2)
        + (p_now[0, 0] - p_now[0, N-2])/(beta*gamma*m2*2*dx)
        )
    flux_array[1, N-1, 0] = (-1.0)*(
        (force2_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m2)
        + (p_now[N-1, 1] - p_now[N-1, N-1])/(beta*gamma*m2*2*dx)
        )
    flux_array[1, N-1, N-1] = (-1.0)*(
        (force2_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m2)
        + (p_now[N-1, 0] - p_now[N-1, N-2])/(beta*gamma*m2*2*dx)
        )

    # for points with well defined neighbours
    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] = (-1.0)*(
            (force1_at_pos[0, i]*p_now[0, i])/(gamma*m1)
            + (p_now[1, i] - p_now[N-1, i])/(beta*gamma*m1*2*dx)
        )
        flux_array[0, i, 0] = (-1.0)*(
            (force1_at_pos[i, 0]*p_now[i, 0])/(gamma*m1)
            + (p_now[i+1, 0]- p_now[i-1, 0])/(beta*gamma*m1*2*dx)
        )

        # second component
        flux_array[1, 0, i] = (-1.0)*(
            (force2_at_pos[0, i]*p_now[0, i])/(gamma*m2)
            + (p_now[0, i+1] - p_now[0, i-1])/(beta*gamma*m2*2*dx)
            )
        flux_array[1, i, 0] = (-1.0)*(
            (force2_at_pos[i, 0]*p_now[i, 0])/(gamma*m2)
            + (p_now[i, 1] - p_now[i, N-1])/(beta*gamma*m2*2*dx)
            )

        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] = (-1.0)*(
                (force1_at_pos[i, j]*p_now[i, j])/(gamma*m1)
                + (p_now[i+1, j] - p_now[i-1, j])/(beta*gamma*m1*2*dx)
                )
            # second component
            flux_array[1, i, j] = (-1.0)*(
                (force2_at_pos[i, j]*p_now[i, j])/(gamma*m2)
                + (p_now[i, j+1] - p_now[i, j-1])/(beta*gamma*m2*2*dx)
                )

        # update rest of edges not corners
        # first component
        flux_array[0, N-1, i] = (-1.0)*(
            (force1_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m1)
            + (p_now[0, i] - p_now[N-2, i])/(beta*gamma*m1*2*dx)
            )
        flux_array[0, i, N-1] = (-1.0)*(
            (force1_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m1)
            + (p_now[i+1, N-1] - p_now[i-1, N-1])/(beta*gamma*m1*2*dx)
            )

        # second component
        flux_array[1, N-1, i] = (-1.0)*(
            (force2_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m2)
            + (p_now[N-1, i+1] - p_now[N-1, i-1])/(beta*gamma*m2*2*dx)
            )
        flux_array[1, i, N-1] = (-1.0)*(
            (force2_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m2)
            + (p_now[i, 0] - p_now[i, N-2])/(beta*gamma*m2*2*dx)
            )

# calculate the derivative of a conditional distribution with respect
# to the variable conditioned on. Assume that variable being marginalized
# is represented on the columns of p_now
cdef void calc_learning_rate_func(
    double[:, :] p_now, double[:] marginal_now, double[:,:] flux_array,
    double[:, :] out_array, int N, double dx
    ) nogil:

    cdef Py_ssize_t i, j

    # boundary condition on derivative
    out_array[0, 0] = flux_array[0,0]*(
        ((p_now[0, 1]/marginal_now[1]) - (p_now[0, N-1]/marginal_now[N-1]))/(2*dx)
        )
    out_array[0, N-1] = flux_array[0, N-1]*(
        ((p_now[0, 0]/marginal_now[0]) - (p_now[0, N-2]/marginal_now[N-2]))/(2*dx)
        )
    out_array[N-1, 0] = flux_array[N-1, 0]*(
        ((p_now[N-1, 1]/marginal_now[0]) - (p_now[N-1, N-1]/marginal_now[N-2]))/(2*dx)
        )
    out_array[N-1, N-1] = flux_array[N-1, N-1]*(
        ((p_now[N-1, 0]/marginal_now[0]) - (p_now[N-1, N-2]/marginal_now[N-2]))/(2*dx)
        )

    # points with well defined neighbors
    for i in range(1, N-1):
        for j in range(1, N-1):
            out_array[i, j] = flux_array[i, j]*(
                ((p_now[i, j+1]/marginal_now[j+1]) - (p_now[i, j-1]/marginal_now[j-1]))/(2*dx)
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