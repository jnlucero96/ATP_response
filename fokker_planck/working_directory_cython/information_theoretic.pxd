# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
cdef double pi
cdef double float32_eps
cdef double float64_eps
cdef double calc_transfer_entropy(None) nogil
cdef double calc_learning_rate(None) nogil
cdef double calc_nostalgia(
    double[:, :], double[:, :],
    double[:], double[:], double[:],
    double[:,:],
    int, double
    ) nogil