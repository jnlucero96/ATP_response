# cython: language_level=3
from libc.math cimport exp, sin, cos, fabs
import numpy as np
cimport numpy as np
from cython import cdivision, boundscheck, wraparound
from sys import stdout
# np.set_printoptions(linewidth=200)

DTYPE = np.longdouble
ctypedef np.longdouble_t DTYPE_t
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
cdef double thresh = 0.00000011920928955078  # threshold = float32 machine eps

def launchpad_coupled(
    int steady_state, double cycles, int N, double write_out,
    double period, double Ax, double Axy, double Ay,
    double dt, double m, double beta, double gamma
    ):

    cdef:
        double dx                                                               = (2*pi)/ N
        double time_check                                                       = dx*m*gamma / (1.5*Ax + 0.5*Ay)
        double Z                                                                = 0.0
        double E                                                                = 0.0
        double work
        double heat
        double mean_flux
        double p_sum
        int    i  # declare general iterator variable
        int    x1  # declare position iterator
        int    x2  # declare position iterator
        double E_after_relax
        double E_0
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] prob      = np.empty((N, N), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] p_now     = np.zeros((N, N), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] p_later   = np.empty((N, N), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=3, negative_indices=False, mode='c'] flux      = np.zeros((2, N, N), dtype=DTYPE)   # initalize cumulative flux as list to keep track of flux at every position
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] positions = np.linspace(0, (2*pi)-dx, N, dtype=DTYPE)

    if dt > time_check:
        print("!!!TIME UNSTABLE!!!\n")

    # calculate the partition function
    for x1 in range(N):
        for x2 in range(N):
            Z += exp(-beta*potential_coupled(positions[x1], positions[x2], Ax, Axy, Ay))

    for x1 in range(N):
        for x2 in range(N):
            prob[x1, x2] = exp(-beta*potential_coupled(positions[x1], positions[x2], Ax, Axy, Ay))/Z
            E += potential_coupled(0.0, x2*dx, Ax, Axy, Ay) * exp(-beta * potential_coupled(0.0, x2*dx, Ax, Axy, Ay))*dx/Z

    # if steady_state:
    #     print("Initializing from steady state...")
    #     init_simulation(prob, p_now, N, beta, gamma, dx, dt, m, period, A, k)
    # else:
    #     print("Initializing from Gibbs-Boltzmann Distribution...")
    #     p_now = prob

    # p_now = prob

    for x1 in range(N):
        for x2 in range(N):
            p_now[x1, x2] = 1.0/(N*N)
            # p_now[x1, x2] = prob[x1, x2]

    E_after_relax = E
    E_0 = E

    print("Running simulation now")
    work, heat, p_sum = run_simulation_coupled(
        p_now, flux, dx, dt, cycles, period, m, gamma, beta, Ax, Axy, Ay, E_after_relax
    )

    return flux, flux.mean(axis=(1,2)), work, heat, p_sum, p_now, prob, positions
    # return flux, work, heat, p_sum, p_now, prob, positions

@cdivision(True)
cdef double force1(
    double position1, double position2, double period,
    double M_tot, double m1, double gamma,
    double Ax, double Axy, double Ay
    ): #need the force to calculate work
    cdef double f1 = (0.5)*(
        Axy*sin(position1-position2)
        + (3*Ax*sin((3*position1)-(2*pi/3)))
        )/(gamma*m1)
    return f1

@cdivision(True)
cdef double force2(
    double position1, double position2, double period,
    double M_tot, double m2, double gamma,
    double Ax, double Axy, double Ay
    ): #need the force to calculate work
    cdef double f2 = (0.5)*(
        (-1.0)*Axy*sin(position1-position2)
        + (3*Ay*sin(3*position2))
        )/(gamma*m2)
    return f2

@cdivision(True)
cdef double potential_coupled(
    double position1, double position2,
    double Ax, double Axy, double Ay
    ): #need the potential in the FPE
    cdef double E = 0.5*(
        Ax*(1-cos((3*position1)-(2*pi/3)))
        + Axy*(1-cos(position1-position2))
        + Ay*(1-cos((3*position2)))
        )
    return E

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef double calc_mean(
    np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] array, int N
    ):
    cdef double mean_val = 0.0
    cdef int i, j
    for i in range(N):
        for j in range(N):
            mean_val += array[i, j]

    return mean_val / N

@boundscheck(False)
@wraparound(False)
cdef double calc_sum(
    np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] array, int N
    ):
    cdef double sum_val = 0.0
    cdef int i, j

    for i in range(N):
        for j in range(N):
            sum_val += array[i, j]

    return sum_val

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef void calc_flux(
    np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c']  p_now,
    np.ndarray[DTYPE_t, ndim=3, negative_indices=False, mode='c']  flux_array,
    double period, double M_tot, double m1, double m2, double gamma, double beta,
    double Ax, double Axy, double Ay, int N, double dx, double dt
    ):

    cdef int i, j

    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] += (-1.0)*(
        force1(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[0, 0]
        + (p_now[1, 0] - p_now[N-1, 0])/(beta*gamma*2*dx)
        )*(dt/dx)
    flux_array[0, 0, N-1] += (-1.0)*(
        force1(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[0, N-1]
        + (p_now[1, N-1] - p_now[N-1, N-1])/(beta*gamma*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, 0] += (-1.0)*(
        force1(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[N-1, 0]
        + (p_now[0, 0] - p_now[N-2, 0])/(beta*gamma*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, N-1] += (-1.0)*(
        force1(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[N-1, N-1]
        + (p_now[0, N-1] - p_now[N-2, N-1])/(beta*gamma*2*dx)
        )*(dt/dx)

    # second component
    flux_array[1, 0, 0] += (-1.0)*(
        force2(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[0, 0]
        + (p_now[0, 1] - p_now[0, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, 0, N-1] += (-1.0)*(
        force2(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[0, N-1]
        + (p_now[0, 0] - p_now[0, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, 0] += (-1.0)*(
        force2(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[N-1, 0]
        + (p_now[N-1, 1] - p_now[N-1, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, N-1] += (-1.0)*(
        force2(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[N-1, N-1]
        + (p_now[N-1, 0] - p_now[N-1, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)

    # for points with well defined neighbours
    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] += (-1.0)*(
        force1(0.0, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[0, i]
        + (p_now[1, i] - p_now[N-1, i])/(beta*gamma*2*dx)
        )*(dt/dx)
        flux_array[0, i, 0] += (-1.0)*(
            force1(i*dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[i, 0]
            + (p_now[i+1, 0]- p_now[i-1, 0])/(beta*gamma*2*dx)
        )*(dt/dx)

        # second component
        flux_array[1, 0, i] += (-1.0)*(
            force2(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[0, i]
            + (p_now[0, i+1] - p_now[0, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, 0] += (-1.0)*(
            force2(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[i, 0]
            + (p_now[i, 1] - p_now[i, N-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)

        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] += (-1.0)*(
                force1(i*dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[i, j]
                + (p_now[i+1, j] - p_now[i-1, j])/(beta*gamma*2*dx)
                )*(dt/dx)
            # second component
            flux_array[1, i, j] += (-1.0)*(
                force2(i*dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[i, j]
                + (p_now[i, j+1] - p_now[i, j-1])/(beta*gamma*m2*2*dx)
                )*(dt/dx)

        # update rest of edges not corners
        # first component
        flux_array[0, N-1, i] += (-1.0)*(
            force1(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[N-1, i]
            + (p_now[0, i] - p_now[N-2, i])/(beta*gamma*2*dx)
            )*(dt/dx)
        flux_array[0, i, N-1] += (-1.0)*(
            force1(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[i, N-1]
            + (p_now[i+1, N-1] - p_now[i-1, N-1])/(beta*gamma*2*dx)
            )*(dt/dx)

        # second component
        flux_array[1, N-1, i] += (-1.0)*(
            force2(-dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[N-1, i]
            + (p_now[N-1, i+1] - p_now[N-1, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, N-1] += (-1.0)*(
            force2(i*dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_now[i, N-1]
            + (p_now[i, 0] - p_now[i, N-2])/(beta*gamma*m2*2*dx)
            )*(dt/dx)

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef (double, double, double) run_simulation_coupled(
    np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] p_now,
    np.ndarray[DTYPE_t, ndim=3, negative_indices=False, mode='c'] flux,
    double dx, double dt, double cycles, double period,
    double m, double gamma, double beta, double Ax,
    double Axy, double Ay, double E_after_relax
    ):

    cdef:
        double      work          = 0.0  # Cumulative work
        double      work_inst     = 0.0  # Instantaneous work
        double      heat          = 0.0  # Cumulative heat
        double      heat_inst     = 0.0  # Instantaneous heat
        double      t             = 0.0  # time
        long        step_counter  = 0
        long        print_counter = 0
        double      E_last        = 0.0
        double      E_change_pot  = 0.0
        int         i, j
        int         N             = p_now.shape[0]
        double      m1            = 1.0
        double      m2            = 1.0
        double      M_tot         = m1 + m2
        np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] p_last   = np.empty((N, N), dtype=DTYPE)

    t += dt #step forwards in t

    while t < (cycles*period+dt):

        for i in range(N):
            for j in range(N):
                p_last[i, j] = p_now[i, j] # save previous distribution
                # reset to zero
                p_now[i, j] = 0.0

        E_last = E_after_relax  #Energy at t-dt is E_last
        E_change_pot = 0.0 #intialize energy of system after potential moved

        for i in range(N):
            for j in range(N):
                E_change_pot += potential_coupled(i*dx, j*dx, Ax, Axy, Ay)*p_last[i, j]

        work += E_change_pot - E_last #add to cumulative work
        work_inst = E_change_pot - E_last #work just in this move

        ## Periodic boundary conditions:
        ## Explicity update FPE for the corners

        p_now[0, 0]=(
            p_last[0, 0]
            + dt*(force1(dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[1, 0] - force1(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-1, 0])/(2.0*dx)
            + dt*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(0.0, dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, 1] - force2(0.0, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, N-1])/(2.0*dx)
            + dt*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[0, N-1]=(
            p_last[0, N-1]
            + dt*(force1(dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[1, N-1] - force1(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-1, N-1])/(2.0*dx)
            + dt*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(0.0, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, 0] - force2(0.0, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, N-2])/(2.0*dx)
            + dt*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[N-1, 0]=(
            p_last[N-1, 0]
            + dt*(force1(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[0, 0] - force1(-2.0*dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-2, 0])/(2.0*dx)
            + dt*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(-dx, dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, 1] - force2(-dx, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, N-1])/(2.0*dx)
            + dt*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[N-1, N-1]=(
            p_last[N-1, N-1]
            + dt*(force1(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[0, N-1] - force1(-2.0*dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-2, N-1])/(2.0*dx)
            + dt*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(-dx, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, 0] - force2(-dx, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, N-2])/(2.0*dx)
            + dt*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
            ) #checked

        # iterate through all the coordinates, not on the corners, for both variables
        for i in range(1, N-1):
            ## Periodic boundary conditions:
            ## Explicitly update FPE for edges not corners
            p_now[0, i]=(
                p_last[0, i]
                + dt*(force1(dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[1, i] - force1(-dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-1, i])/(2.0*dx)
                + dt*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(0.0, i*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, i+1] - force2(0.0, i*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, i-1])/(2.0*dx)
                + dt*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
                ) # checked
            p_now[i, 0]=(
                p_last[i, 0]
                + dt*(force1(i*dx+dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i+1, 0] - force1(i*dx-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i-1, 0])/(2.0*dx)
                + dt*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(i*dx, dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, 1] - force2(i*dx, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, N-1])/(2.0*dx)
                + dt*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

            ## all points with well defined neighbours go like so:
            for j in range(1, N-1):
                p_now[i, j]= (
                    p_last[i, j]
                    + dt*(force1(i*dx+dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i+1, j] - force1(i*dx-dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i-1, j])/(2.0*dx)
                    + dt*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                    + dt*(force2(i*dx, j*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, j+1] - force2(i*dx, j*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, j-1])/(2.0*dx)
                    + dt*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                    ) # checked

            ## Explicitly update FPE for rest of edges not corners
            p_now[N-1, i]=(
                p_last[N-1, i]
                + dt*(force1(0.0, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[0, i] - force1(-2.0*dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-2, i])/(2.0*dx)
                + dt*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(-dx, i*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, i+1] - force2(-dx, i*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, i-1])/(2.0*dx)
                + dt*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
                ) # checked
            p_now[i, N-1]=(
                p_last[i, N-1]
                + dt*(force1(i*dx+dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i+1, N-1] - force1(i*dx-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i-1, N-1])/(2.0*dx)
                + dt*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(i*dx, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, 0] - force2(i*dx, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, N-2])/(2.0*dx)
                + dt*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
                ) # checked

        E_after_relax = 0.0 #intialize energy after relax, ie evolving probability distribution
        for i in range(N):
            for j in range(N):
                E_after_relax += potential_coupled(i*dx, j*dx, Ax, Axy, Ay)*p_now[i, j]

        heat += E_after_relax - E_change_pot #adds to cumulative heat
        heat_inst = E_after_relax - E_change_pot #heat just in this move

        calc_flux(p_now, flux, period, M_tot, m1, m2, gamma, beta, Ax, Axy, Ay, N, dx, dt)

        t += dt

    return work/cycles, heat/cycles, calc_sum(p_now, N)