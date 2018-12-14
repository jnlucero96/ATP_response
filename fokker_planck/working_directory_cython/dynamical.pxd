# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
cdef void calc_flux(
    double[:], double[:, :], double[:, :, :], double[:, :], double[:, :],
    double, double, double, double, int, double, double
    ) nogil