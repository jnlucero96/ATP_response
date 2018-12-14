# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False

cdef double pi
cdef double force1(double, double, double, double, double) nogil # force on system X
cdef double force2(double, double, double, double, double) nogil # force on system Y
cdef double potential(double, double, double, double, double) nogil