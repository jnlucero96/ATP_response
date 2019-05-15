# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport exp, fabs, log, sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
# float64 machine eps
cdef double float64_eps = 2.22044604925031308084726e-16

def calc_flux_1d(
    double[:] positions, double[:] p_now,
    double[:] drift_at_pos, double[:] diffusion_at_pos,
    double[:] flux_array,
    int N, double dx
    ):

    calc_flux_func_1d(
        positions, p_now, drift_at_pos, diffusion_at_pos, flux_array, N, dx
        )

cdef void calc_flux_func_1d(
    double[:] positions, double[:] p_now,
    double[:] drift_at_pos, double[:] diffusion_at_pos,
    double[:] flux_array,
    int N, double dx
    ) nogil:

    cdef Py_ssize_t i,j

    # periodic boundary conditions:
    # explicit update of the end points 
    flux_array[0] = (
        (drift_at_pos[0]*p_now[0])
        -(diffusion_at_pos[1]*p_now[1]-diffusion_at_pos[N-1]*p_now[N-1])/(2.0*dx)
        )

    flux_array[N-1] = (
        (drift_at_pos[N-1]*p_now[N-1])
        -(diffusion_at_pos[0]*p_now[0]-diffusion_at_pos[N-2]*p_now[N-2])/(2.0*dx)
        )

    # for points with well defined neighbours
    for i in range(1, N-1):
        flux_array[i] = ( 
            (drift_at_pos[i]*p_now[i])
            -(diffusion_at_pos[i+1]*p_now[i+1]-diffusion_at_pos[i-1]*p_now[i-1])/(2.0*dx)
        )