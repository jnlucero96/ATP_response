# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport log

cdef double calc_transfer_entropy(None) nogil:
    # first compute H(Y_t|Y_{t-1:t-L}), unsure how to compute
    # next compute H(Y_t|Y_{t-1:t-L}, X_{t-1:t-L}), unsure how to compute
    # compute the transfer entropy by subtraction
    return 0.0

cdef double calc_learning_rate(None) nogil:
    # first compute I(X_{t};Y_{t+\tau})
    # then compute derivative wrt \tau
    return 0.0

cdef double calc_nostalgia(
    double[:, :] p_xy_now, double[:, :] p_x_now_y_next,
    double[:] p_x_now, double[:] p_y_now, double[:] p_y_next,
    double[:, :] p_next,
    int N, double dx
    ) nogil:

    cdef:
        double I_mem  = 0.0
        double I_pred = 0.0

        # declare iterator variables
        Py_ssize_t i, j

    # compute I_{mem} = I(X_{t}, Y_{t}), assume dx = dy
    for i in range(N):
        for j in range(N):
            I_mem += p_xy_now[i, j]*log(p_xy_now[i, j]/(p_x_now[i]*p_y_now[j]))
    I_mem *= (dx*dx)

    # compute I_{pred} = I(X_{t}, Y_{t+1})
    for i in range(N):
        for j in range(N):
            I_pred += p_x_now_y_next[i, j]*log(p_x_now_y_next[i, j]/(p_x_now[i]*p_y_next[j]))
    I_pred *= (dx*dx)

    # compute the nostalgia by subtraction
    return I_mem - I_pred