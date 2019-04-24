# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, log, sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

def calc_flux_1d(
    double[:] positions, double[:] p_now,
    double[:] force_at_pos,
    double[:] flux_array,
    double m, double gamma, double beta, int N, double dx, double dt
    ):

    calc_flux_func_1d(
        positions, p_now, force_at_pos, flux_array, m, gamma, beta, N, dx, dt
        )

cdef void calc_flux_func_1d(
    double[:] positions, double[:] p_now,
    double[:] force_at_pos,
    double[:] flux_array,
    double m, double gamma, double beta, int N, double dx, double dt
    ) nogil:

    cdef Py_ssize_t i,j

    # explicit update of the ends
    flux_array[0] = (-1.0)*(
        (force_at_pos[0]*p_now[0])/(gamma*m)
        + (p_now[1] - p_now[N-1])/(beta*gamma*m*2.0*dx)
        )

    flux_array[N-1] = (-1.0)*(
        (force_at_pos[N-1]*p_now[N-1])/(gamma*m)
        + (p_now[0] - p_now[N-2])/(beta*gamma*m*2.0*dx)
        )

    # for points with well defined neighbours
    for i in range(1, N-1):
        # explicitly update for
        flux_array[i] = (-1.0)*(
            (force_at_pos[i]*p_now[i])/(gamma*m)
            + (p_now[i+1] - p_now[i-1])/(beta*gamma*m*2.0*dx)
        )