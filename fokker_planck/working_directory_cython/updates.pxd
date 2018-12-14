# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
cdef void update_probability_x_half(
    double[:], double[:, :], double[:, :], double[:, :],
    double, double, double, int, double, double
    ) nogil
cdef void update_probability_y_half(
    double[:], double[:, :], double[:, :], double[:, :],
    double, double, double, int, double, double
    ) nogil
cdef void update_probability_t_half(double[:, :], double[:, :], int) nogil
cdef void update_probability_x(
    double[:], double[:, :], double[:, :], double[:, :],
    double, double, double, int, double, double
    ) nogil
cdef void update_probability_y(
    double[:], double[:, :], double[:, :], double[:, :],
    double, double, double, int, double, double
    ) nogil
cdef void update_probability_t(double[:, :], double[:, :], int) nogil
cdef void update_probability_full(
    double[:], double[:, :], double[:, :], double[:, :], double[:, :], double,
    double, double, double, int, double, double
    ) nogil