# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False

###############################################################################
###############################################################################
############
############ UPDATE STEPS - HALF UPDATES
############
###############################################################################
###############################################################################

cdef void update_probability_x_half(
    double[:] positions,
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
    p_now[0, 0]=(
        p_last[0, 0]
        + (dt/2.0)*(force1_at_pos[1, 0]*p_last[1, 0]-force1_at_pos[N-1, 0]*p_last[N-1, 0])/(gamma*m1*2.0*dx)
        + (dt/2.0)*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
        ) # checked
    p_now[0, N-1]=(
        p_last[0, N-1]
        + (dt/2.0)*(force1_at_pos[1, N-1]*p_last[1, N-1]-force1_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m1*2.0*dx)
        + (dt/2.0)*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
        ) # checked
    p_now[N-1, 0]=(
        p_last[N-1, 0]
        + (dt/2.0)*(force1_at_pos[0, 0]*p_last[0, 0]-force1_at_pos[N-2, 0]*p_last[N-2, 0])/(gamma*m1*2.0*dx)
        + (dt/2.0)*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
        ) # checked
    p_now[N-1, N-1]=(
        p_last[N-1, N-1]
        + (dt/2.0)*(force1_at_pos[0, N-1]*p_last[0, N-1]-force1_at_pos[N-2, N-1]*p_last[N-2, N-1])/(gamma*m1*2.0*dx)
        + (dt/2.0)*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i]=(
            p_last[0, i]
            + (dt/2.0)*(force1_at_pos[1, i]*p_last[1, i]-force1_at_pos[N-1, i]*p_last[N-1, i])/(gamma*m1*2.0*dx)
            + (dt/2.0)*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
            ) # checked
        p_now[i, 0]=(
            p_last[i, 0]
            + (dt/2.0)*(force1_at_pos[i+1, 0]*p_last[i+1, 0]-force1_at_pos[i-1, 0]*p_last[i-1, 0])/(gamma*m1*2.0*dx)
            + (dt/2.0)*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j]= (
                p_last[i, j]
                + (dt/2.0)*(force1_at_pos[i+1, j]*p_last[i+1, j]-force1_at_pos[i-1, j]*p_last[i-1, j])/(gamma*m1*2.0*dx)
                + (dt/2.0)*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i]=(
            p_last[N-1, i]
            + (dt/2.0)*(force1_at_pos[0, i]*p_last[0, i]-force1_at_pos[N-2, i]*p_last[N-2, i])/(gamma*m1*2.0*dx)
            + (dt/2.0)*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
            ) # checked
        p_now[i, N-1]=(
            p_last[i, N-1]
            + (dt/2.0)*(force1_at_pos[i+1, N-1]*p_last[i+1, N-1]-force1_at_pos[i-1, N-1]*p_last[i-1, N-1])/(gamma*m1*2.0*dx)
            + (dt/2.0)*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
            ) # checked

cdef void update_probability_y_half(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] force2_at_pos,
    double m2, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0]=(
            p_last[0, 0]
            + (dt/2.0)*(force2_at_pos[0, 1]*p_last[0, 1]-force2_at_pos[0, N-1]*p_last[0, N-1])/(gamma*m2*2.0*dx)
            + (dt/2.0)*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
    p_now[0, N-1]=(
        p_last[0, N-1]
        + (dt/2.0)*(force2_at_pos[0, 0]*p_last[0, 0]-force2_at_pos[0, N-2]*p_last[0, N-2])/(gamma*m2*2.0*dx)
        + (dt/2.0)*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, 0]=(
        p_last[N-1, 0]
        + (dt/2.0)*(force2_at_pos[N-1, 1]*p_last[N-1, 1]-force2_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m2*2.0*dx)
        + (dt/2.0)*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, N-1]=(
        p_last[N-1, N-1]
        + (dt/2.0)*(force2_at_pos[N-1, 0]*p_last[N-1, 0]-force2_at_pos[N-1, N-2]*p_last[N-1, N-2])/(gamma*m2*2.0*dx)
        + (dt/2.0)*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i]=(
            p_last[0, i]
            + (dt/2.0)*(force2_at_pos[0, i+1]*p_last[0, i+1]-force2_at_pos[0, i-1]*p_last[0, i-1])/(gamma*m2*2.0*dx)
            + (dt/2.0)*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, 0]=(
            p_last[i, 0]
            + (dt/2.0)*(force2_at_pos[i, 1]*p_last[i, 1]-force2_at_pos[i, N-1]*p_last[i, N-1])/(gamma*m2*2.0*dx)
            + (dt/2.0)*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j]= (
                p_last[i, j]
                + (dt/2.0)*(force2_at_pos[i, j+1]*p_last[i, j+1]-force2_at_pos[i, j-1]*p_last[i, j-1])/(gamma*m2*2.0*dx)
                + (dt/2.0)*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i]=(
            p_last[N-1, i]
            + (dt/2.0)*(force2_at_pos[N-1, i+1]*p_last[N-1, i+1]-force2_at_pos[N-1, i-1]*p_last[N-1, i-1])/(gamma*m2*2.0*dx)
            + (dt/2.0)*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, N-1]=(
            p_last[i, N-1]
            + (dt/2.0)*(force2_at_pos[i, 0]*p_last[i, 0]-force2_at_pos[i, N-2]*p_last[i, N-2])/(gamma*m2*2.0*dx)
            + (dt/2.0)*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked

cdef void update_probability_t_half(double[:, :] p_now, double[:, :] p_last, int N) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0] = p_last[0, 0] / 2.0 # checked
    p_now[0, N-1] = p_last[0, N-1] / 2.0 # checked
    p_now[N-1, 0] = p_last[N-1, 0] / 2.0 # checked
    p_now[N-1, N-1] = p_last[N-1, N-1] / 2.0 #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i] = p_last[0, i] / 2.0# checked
        p_now[i, 0] = p_last[i, 0] / 2.0

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = p_last[i, j] / 2.0 # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i] = p_last[N-1, i] / 2.0 # checked
        p_now[i, N-1] = p_last[i, N-1] / 2.0 # checked

###############################################################################
###############################################################################
############
############ UPDATE STEPS - FULL UPDATES
############
###############################################################################
###############################################################################

cdef void update_probability_x(
    double[:] positions,
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
    p_now[0, 0]=(
            p_last[0, 0]
            + dt*(force1_at_pos[1, 0]*p_last[1, 0]-force1_at_pos[N-1, 0]*p_last[N-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
            ) # checked
    p_now[0, N-1]=(
        p_last[0, N-1]
        + dt*(force1_at_pos[1, N-1]*p_last[1, N-1]-force1_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
        ) # checked
    p_now[N-1, 0]=(
        p_last[N-1, 0]
        + dt*(force1_at_pos[0, 0]*p_last[0, 0]-force1_at_pos[N-2, 0]*p_last[N-2, 0])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
        ) # checked
    p_now[N-1, N-1]=(
        p_last[N-1, N-1]
        + dt*(force1_at_pos[0, N-1]*p_last[0, N-1]-force1_at_pos[N-2, N-1]*p_last[N-2, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i]=(
            p_last[0, i]
            + dt*(force1_at_pos[1, i]*p_last[1, i]-force1_at_pos[N-1, i]*p_last[N-1, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
            ) # checked
        p_now[i, 0]=(
            p_last[i, 0]
            + dt*(force1_at_pos[i+1, 0]*p_last[i+1, 0]-force1_at_pos[i-1, 0]*p_last[i-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j]= (
                p_last[i, j]
                + dt*(force1_at_pos[i+1, j]*p_last[i+1, j]-force1_at_pos[i-1, j]*p_last[i-1, j])/(gamma*m1*2.0*dx)
                + dt*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i]=(
            p_last[N-1, i]
            + dt*(force1_at_pos[0, i]*p_last[0, i]-force1_at_pos[N-2, i]*p_last[N-2, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
            ) # checked
        p_now[i, N-1]=(
            p_last[i, N-1]
            + dt*(force1_at_pos[i+1, N-1]*p_last[i+1, N-1]-force1_at_pos[i-1, N-1]*p_last[i-1, N-1])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
            ) # checked

cdef void update_probability_y(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] force2_at_pos,
    double m2, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0]=(
            p_last[0, 0]
            + dt*(force2_at_pos[0, 1]*p_last[0, 1]-force2_at_pos[0, N-1]*p_last[0, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
    p_now[0, N-1]=(
        p_last[0, N-1]
        + dt*(force2_at_pos[0, 0]*p_last[0, 0]-force2_at_pos[0, N-2]*p_last[0, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, 0]=(
        p_last[N-1, 0]
        + dt*(force2_at_pos[N-1, 1]*p_last[N-1, 1]-force2_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, N-1]=(
        p_last[N-1, N-1]
        + dt*(force2_at_pos[N-1, 0]*p_last[N-1, 0]-force2_at_pos[N-1, N-2]*p_last[N-1, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i]=(
            p_last[0, i]
            + dt*(force2_at_pos[0, i+1]*p_last[0, i+1]-force2_at_pos[0, i-1]*p_last[0, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, 0]=(
            p_last[i, 0]
            + dt*(force2_at_pos[i, 1]*p_last[i, 1]-force2_at_pos[i, N-1]*p_last[i, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j]= (
                p_last[i, j]
                + dt*(force2_at_pos[i, j+1]*p_last[i, j+1]-force2_at_pos[i, j-1]*p_last[i, j-1])/(gamma*m2*2.0*dx)
                + dt*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i]=(
            p_last[N-1, i]
            + dt*(force2_at_pos[N-1, i+1]*p_last[N-1, i+1]-force2_at_pos[N-1, i-1]*p_last[N-1, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, N-1]=(
            p_last[i, N-1]
            + dt*(force2_at_pos[i, 0]*p_last[i, 0]-force2_at_pos[i, N-2]*p_last[i, N-2])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked

cdef void update_probability_t(double[:, :] p_now, double[:, :] p_last, int N) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0] = p_last[0, 0] # checked
    p_now[0, N-1] = p_last[0, N-1] # checked
    p_now[N-1, 0] = p_last[N-1, 0] # checked
    p_now[N-1, N-1] = p_last[N-1, N-1] #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i] = p_last[0, i] # checked
        p_now[i, 0] = p_last[i, 0] # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j] = p_last[i, j] # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i] = p_last[N-1, i] # checked
        p_now[i, N-1] = p_last[i, N-1] # checked

cdef void update_probability_full(
    double[:] positions,
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :] force1_at_pos,
    double[:, :] force2_at_pos,
    double m1, double m2, double gamma, double beta,
    int N, double dx, double dt
    ) nogil:

    # declare iterator variables
    cdef Py_ssize_t i, j

    ## Periodic boundary conditions:
    ## Explicity update FPE for the corners
    p_now[0, 0]=(
            p_last[0, 0]
            + dt*(force1_at_pos[1, 0]*p_last[1, 0]-force1_at_pos[N-1, 0]*p_last[N-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[0, 1]*p_last[0, 1]-force2_at_pos[0, N-1]*p_last[0, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
    p_now[0, N-1]=(
        p_last[0, N-1]
        + dt*(force1_at_pos[1, N-1]*p_last[1, N-1]-force1_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
        + dt*(force2_at_pos[0, 0]*p_last[0, 0]-force2_at_pos[0, N-2]*p_last[0, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, 0]=(
        p_last[N-1, 0]
        + dt*(force1_at_pos[0, 0]*p_last[0, 0]-force1_at_pos[N-2, 0]*p_last[N-2, 0])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
        + dt*(force2_at_pos[N-1, 1]*p_last[N-1, 1]-force2_at_pos[N-1, N-1]*p_last[N-1, N-1])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
        ) # checked
    p_now[N-1, N-1]=(
        p_last[N-1, N-1]
        + dt*(force1_at_pos[0, N-1]*p_last[0, N-1]-force1_at_pos[N-2, N-1]*p_last[N-2, N-1])/(gamma*m1*2.0*dx)
        + dt*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
        + dt*(force2_at_pos[N-1, 0]*p_last[N-1, 0]-force2_at_pos[N-1, N-2]*p_last[N-1, N-2])/(gamma*m2*2.0*dx)
        + dt*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
        ) #checked

    # iterate through all the coordinates, not on the corners, for both variables
    for i in range(1, N-1):
        ## Periodic boundary conditions:
        ## Explicitly update FPE for edges not corners
        p_now[0, i]=(
            p_last[0, i]
            + dt*(force1_at_pos[1, i]*p_last[1, i]-force1_at_pos[N-1, i]*p_last[N-1, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[0, i+1]*p_last[0, i+1]-force2_at_pos[0, i-1]*p_last[0, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, 0]=(
            p_last[i, 0]
            + dt*(force1_at_pos[i+1, 0]*p_last[i+1, 0]-force1_at_pos[i-1, 0]*p_last[i-1, 0])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[i, 1]*p_last[i, 1]-force2_at_pos[i, N-1]*p_last[i, N-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked

        ## all points with well defined neighbours go like so:
        for j in range(1, N-1):
            p_now[i, j]= (
                p_last[i, j]
                + dt*(force1_at_pos[i+1, j]*p_last[i+1, j]-force1_at_pos[i-1, j]*p_last[i-1, j])/(gamma*m1*2.0*dx)
                + dt*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                + dt*(force2_at_pos[i, j+1]*p_last[i, j+1]-force2_at_pos[i, j-1]*p_last[i, j-1])/(gamma*m2*2.0*dx)
                + dt*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

        ## Explicitly update FPE for rest of edges not corners
        p_now[N-1, i]=(
            p_last[N-1, i]
            + dt*(force1_at_pos[0, i]*p_last[0, i]-force1_at_pos[N-2, i]*p_last[N-2, i])/(gamma*m1*2.0*dx)
            + dt*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[N-1, i+1]*p_last[N-1, i+1]-force2_at_pos[N-1, i-1]*p_last[N-1, i-1])/(gamma*m2*2.0*dx)
            + dt*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[i, N-1]=(
            p_last[i, N-1]
            + dt*(force1_at_pos[i+1, N-1]*p_last[i+1, N-1]-force1_at_pos[i-1, N-1]*p_last[i-1, N-1])/(gamma*m1*2.0*dx)
            + dt*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2_at_pos[i, 0]*p_last[i, 0]-force2_at_pos[i, N-2]*p_last[i, N-2])/(gamma*m2*2.0*dx)
            + dt*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked