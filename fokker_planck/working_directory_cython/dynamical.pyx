# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
cdef void calc_flux(
    double[:] positions,
    double[:, :]  p_now,
    double[:, :, :]  flux_array,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    double m1, double m2, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    cdef Py_ssize_t i, j # declare iterator variables

    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] += (-1.0)*(
        (force1_at_pos[0, 0]*p_now[0, 0])/(gamma*m1)
        + (p_now[1, 0] - p_now[N-1, 0])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
    flux_array[0, 0, N-1] += (-1.0)*(
        (force1_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m1)
        + (p_now[1, N-1] - p_now[N-1, N-1])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, 0] += (-1.0)*(
        (force1_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m1)
        + (p_now[0, 0] - p_now[N-2, 0])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, N-1] += (-1.0)*(
        (force1_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m1)
        + (p_now[0, N-1] - p_now[N-2, N-1])/(beta*gamma*m1*2*dx)
        )*(dt/dx)

    # second component
    flux_array[1, 0, 0] += (-1.0)*(
        (force2_at_pos[0, 0]*p_now[0, 0])/(gamma*m2)
        + (p_now[0, 1] - p_now[0, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, 0, N-1] += (-1.0)*(
        (force2_at_pos[0, N-1]*p_now[0, N-1])/(gamma*m2)
        + (p_now[0, 0] - p_now[0, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, 0] += (-1.0)*(
        (force2_at_pos[N-1, 0]*p_now[N-1, 0])/(gamma*m2)
        + (p_now[N-1, 1] - p_now[N-1, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, N-1] += (-1.0)*(
        (force2_at_pos[N-1, N-1]*p_now[N-1, N-1])/(gamma*m2)
        + (p_now[N-1, 0] - p_now[N-1, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)

    # for points with well defined neighbours
    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] += (-1.0)*(
            (force1_at_pos[0, i]*p_now[0, i])/(gamma*m1)
            + (p_now[1, i] - p_now[N-1, i])/(beta*gamma*m1*2*dx)
        )*(dt/dx)
        flux_array[0, i, 0] += (-1.0)*(
            (force1_at_pos[i, 0]*p_now[i, 0])/(gamma*m1)
            + (p_now[i+1, 0]- p_now[i-1, 0])/(beta*gamma*m1*2*dx)
        )*(dt/dx)

        # second component
        flux_array[1, 0, i] += (-1.0)*(
            (force2_at_pos[0, i]*p_now[0, i])/(gamma*m2)
            + (p_now[0, i+1] - p_now[0, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, 0] += (-1.0)*(
            (force2_at_pos[i, 0]*p_now[i, 0])/(gamma*m2)
            + (p_now[i, 1] - p_now[i, N-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)

        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] += (-1.0)*(
                (force1_at_pos[i, j]*p_now[i, j])/(gamma*m1)
                + (p_now[i+1, j] - p_now[i-1, j])/(beta*gamma*m1*2*dx)
                )*(dt/dx)
            # second component
            flux_array[1, i, j] += (-1.0)*(
                (force2_at_pos[i, j]*p_now[i, j])/(gamma*m2)
                + (p_now[i, j+1] - p_now[i, j-1])/(beta*gamma*m2*2*dx)
                )*(dt/dx)

        # update rest of edges not corners
        # first component
        flux_array[0, N-1, i] += (-1.0)*(
            (force1_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m1)
            + (p_now[0, i] - p_now[N-2, i])/(beta*gamma*m1*2*dx)
            )*(dt/dx)
        flux_array[0, i, N-1] += (-1.0)*(
            (force1_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m1)
            + (p_now[i+1, N-1] - p_now[i-1, N-1])/(beta*gamma*m1*2*dx)
            )*(dt/dx)

        # second component
        flux_array[1, N-1, i] += (-1.0)*(
            (force2_at_pos[N-1, i]*p_now[N-1, i])/(gamma*m2)
            + (p_now[N-1, i+1] - p_now[N-1, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, N-1] += (-1.0)*(
            (force2_at_pos[i, N-1]*p_now[i, N-1])/(gamma*m2)
            + (p_now[i, 0] - p_now[i, N-2])/(beta*gamma*m2*2*dx)
            )*(dt/dx)