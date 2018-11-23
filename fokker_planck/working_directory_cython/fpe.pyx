# cython: language_level=3
from libc.math cimport exp, sin, cos, fabs
import numpy as np
cimport numpy as np
from cython import cdivision, boundscheck, wraparound

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582
# float32 machine eps
cdef double float32_eps = 0.00000011920928955078

@boundscheck(False)
@wraparound(False)
def launchpad_coupled(
    double[:, :] prob, double[:, :] p_now, double[:, :] p_last,
    double[:, :, :] flux, double[:] positions, int N,
    double dx, double time_check, int steady_state, double cycles,
    double period, double Ax, double Axy, double Ay, double H, double A,
    double dt, double m, double beta, double gamma
    ):

    cdef:
        double Z = 0.0
        double E = 0.0
        double work
        double heat
        double mean_flux
        double p_sum
        int    i  # declare general iterator variable
        int    x1  # declare position iterator
        int    x2  # declare position iterator
        double E_after_relax
        double E_0

    if dt > time_check:
        print("!!!TIME UNSTABLE!!!\n")

    # calculate the partition function
    for x1 in range(N):
        for x2 in range(N):
            Z += exp(-beta*potential(positions[x1], positions[x2], Ax, Axy, Ay))

    for x1 in range(N):
        for x2 in range(N):
            prob[x1, x2] = exp(-beta*potential(positions[x1], positions[x2], Ax, Axy, Ay))*(Z**(-1.0))
            E += potential(0.0, x2*dx, Ax, Axy, Ay) * exp(-beta * potential(0.0, x2*dx, Ax, Axy, Ay))*dx*(Z**(-1.0))

    # initialize the simulation from the equilibrium distribution
    for x1 in range(N):
        for x2 in range(N):
            p_now[x1, x2] = prob[x1, x2]

    E_after_relax = E
    E_0 = E

    work, heat = run_simulation_coupled(
        p_now, p_last, flux, dx, dt, cycles, period,
        m, gamma, beta, Ax, Axy, Ay, H, A, E_after_relax
    )

    return work, heat

@cdivision(True)
cdef double force1(
    double position1, double position2, double period,
    double M_tot, double m1, double gamma,
    double Ax, double Axy, double Ay, double H
    ) nogil: # force on system X
    cdef double f1 = (0.5)*(
        Axy*sin(position1-position2)
        + (3*Ax*sin((3*position1)-(2*pi/3)))
        - H)/(gamma*m1)
    return f1

@cdivision(True)
cdef double force2(
    double position1, double position2, double period,
    double M_tot, double m2, double gamma,
    double Ax, double Axy, double Ay, double A
    ) nogil: # force on system Y
    cdef double f2 = (0.5)*(
        (-1.0)*Axy*sin(position1-position2)
        + (3*Ay*sin(3*position2))
        + A)/(gamma*m2)
    return f2

@cdivision(True)
cdef double potential(
    double position1, double position2,
    double Ax, double Axy, double Ay
    ) nogil: #need the potential in the FPE
    cdef double E = 0.5*(
        Ax*(1-cos((3*position1)-(2*pi/3)))
        + Axy*(1-cos(position1-position2))
        + Ay*(1-cos((3*position2)))
        )
    return E

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef void calc_flux(
    double[:, :]  p_now,
    double[:, :, :]  flux_array,
    double period, double M_tot, double m1, double m2, double gamma, double beta,
    double Ax, double Axy, double Ay, double H, double A, int N, double dx, double dt
    ) nogil:

    cdef int i, j

    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] += (-1.0)*(
        force1(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[0, 0]
        + (p_now[1, 0] - p_now[N-1, 0])/(beta*gamma*2*dx)
        )*(dt/dx)
    flux_array[0, 0, N-1] += (-1.0)*(
        force1(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[0, N-1]
        + (p_now[1, N-1] - p_now[N-1, N-1])/(beta*gamma*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, 0] += (-1.0)*(
        force1(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[N-1, 0]
        + (p_now[0, 0] - p_now[N-2, 0])/(beta*gamma*2*dx)
        )*(dt/dx)
    flux_array[0, N-1, N-1] += (-1.0)*(
        force1(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[N-1, N-1]
        + (p_now[0, N-1] - p_now[N-2, N-1])/(beta*gamma*2*dx)
        )*(dt/dx)

    # second component
    flux_array[1, 0, 0] += (-1.0)*(
        force2(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[0, 0]
        + (p_now[0, 1] - p_now[0, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, 0, N-1] += (-1.0)*(
        force2(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[0, N-1]
        + (p_now[0, 0] - p_now[0, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, 0] += (-1.0)*(
        force2(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[N-1, 0]
        + (p_now[N-1, 1] - p_now[N-1, N-1])/(beta*gamma*m2*2*dx)
        )*(dt/dx)
    flux_array[1, N-1, N-1] += (-1.0)*(
        force2(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[N-1, N-1]
        + (p_now[N-1, 0] - p_now[N-1, N-2])/(beta*gamma*m2*2*dx)
        )*(dt/dx)

    # for points with well defined neighbours
    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] += (-1.0)*(
            force1(0.0, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[0, i]
            + (p_now[1, i] - p_now[N-1, i])/(beta*gamma*2*dx)
        )*(dt/dx)
        flux_array[0, i, 0] += (-1.0)*(
            force1(i*dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[i, 0]
            + (p_now[i+1, 0]- p_now[i-1, 0])/(beta*gamma*2*dx)
        )*(dt/dx)

        # second component
        flux_array[1, 0, i] += (-1.0)*(
            force2(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[0, i]
            + (p_now[0, i+1] - p_now[0, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, 0] += (-1.0)*(
            force2(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[i, 0]
            + (p_now[i, 1] - p_now[i, N-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)

        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] += (-1.0)*(
                force1(i*dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[i, j]
                + (p_now[i+1, j] - p_now[i-1, j])/(beta*gamma*2*dx)
                )*(dt/dx)
            # second component
            flux_array[1, i, j] += (-1.0)*(
                force2(i*dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[i, j]
                + (p_now[i, j+1] - p_now[i, j-1])/(beta*gamma*m2*2*dx)
                )*(dt/dx)

        # update rest of edges not corners
        # first component
        flux_array[0, N-1, i] += (-1.0)*(
            force1(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[N-1, i]
            + (p_now[0, i] - p_now[N-2, i])/(beta*gamma*2*dx)
            )*(dt/dx)
        flux_array[0, i, N-1] += (-1.0)*(
            force1(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_now[i, N-1]
            + (p_now[i+1, N-1] - p_now[i-1, N-1])/(beta*gamma*2*dx)
            )*(dt/dx)

        # second component
        flux_array[1, N-1, i] += (-1.0)*(
            force2(-dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[N-1, i]
            + (p_now[N-1, i+1] - p_now[N-1, i-1])/(beta*gamma*m2*2*dx)
            )*(dt/dx)
        flux_array[1, i, N-1] += (-1.0)*(
            force2(i*dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, A)*p_now[i, N-1]
            + (p_now[i, 0] - p_now[i, N-2])/(beta*gamma*m2*2*dx)
            )*(dt/dx)

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef (double, double) run_simulation_coupled(
    double[:, :] p_now,
    double[:, :] p_last,
    double[:, :, :] flux,
    double dx, double dt, double cycles, double period,
    double m, double gamma, double beta, double Ax,
    double Axy, double Ay, double H, double A, double E_after_relax
    ) nogil:

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

    t += dt #step forwards in t

    while t < (cycles*period+dt):

        # save previous distribution
        for i in range(N):
            for j in range(N):
                p_last[i, j] = p_now[i, j]

        # reset to zero
        for i in range(N):
            for j in range(N):
                p_now[i, j] = 0.0

        E_last = E_after_relax  #Energy at t-dt is E_last
        E_change_pot = 0.0 #intialize energy of system after potential moved

        for i in range(N):
            for j in range(N):
                E_change_pot += potential(i*dx, j*dx, Ax, Axy, Ay)*p_last[i, j]

        work += E_change_pot - E_last #add to cumulative work
        work_inst = E_change_pot - E_last #work just in this move

        ## Periodic boundary conditions:
        ## Explicity update FPE for the corners

        p_now[0, 0]=(
            p_last[0, 0]
            + dt*(force1(dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[1, 0] - force1(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[N-1, 0])/(2.0*dx)
            + dt*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(0.0, dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[0, 1] - force2(0.0, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[0, N-1])/(2.0*dx)
            + dt*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[0, N-1]=(
            p_last[0, N-1]
            + dt*(force1(dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[1, N-1] - force1(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[N-1, N-1])/(2.0*dx)
            + dt*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(0.0, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[0, 0] - force2(0.0, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[0, N-2])/(2.0*dx)
            + dt*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[N-1, 0]=(
            p_last[N-1, 0]
            + dt*(force1(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[0, 0] - force1(-2.0*dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[N-2, 0])/(2.0*dx)
            + dt*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(-dx, dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[N-1, 1] - force2(-dx, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[N-1, N-1])/(2.0*dx)
            + dt*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[N-1, N-1]=(
            p_last[N-1, N-1]
            + dt*(force1(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[0, N-1] - force1(-2.0*dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[N-2, N-1])/(2.0*dx)
            + dt*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(-dx, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[N-1, 0] - force2(-dx, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[N-1, N-2])/(2.0*dx)
            + dt*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
            ) #checked

        # iterate through all the coordinates, not on the corners, for both variables
        for i in range(1, N-1):
            ## Periodic boundary conditions:
            ## Explicitly update FPE for edges not corners
            p_now[0, i]=(
                p_last[0, i]
                + dt*(force1(dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[1, i] - force1(-dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[N-1, i])/(2.0*dx)
                + dt*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(0.0, i*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[0, i+1] - force2(0.0, i*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[0, i-1])/(2.0*dx)
                + dt*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
                ) # checked
            p_now[i, 0]=(
                p_last[i, 0]
                + dt*(force1(i*dx+dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[i+1, 0] - force1(i*dx-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[i-1, 0])/(2.0*dx)
                + dt*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(i*dx, dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[i, 1] - force2(i*dx, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[i, N-1])/(2.0*dx)
                + dt*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

            ## all points with well defined neighbours go like so:
            for j in range(1, N-1):
                p_now[i, j]= (
                    p_last[i, j]
                    + dt*(force1(i*dx+dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[i+1, j] - force1(i*dx-dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[i-1, j])/(2.0*dx)
                    + dt*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                    + dt*(force2(i*dx, j*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[i, j+1] - force2(i*dx, j*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[i, j-1])/(2.0*dx)
                    + dt*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                    ) # checked

            ## Explicitly update FPE for rest of edges not corners
            p_now[N-1, i]=(
                p_last[N-1, i]
                + dt*(force1(0.0, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[0, i] - force1(-2.0*dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[N-2, i])/(2.0*dx)
                + dt*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(-dx, i*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[N-1, i+1] - force2(-dx, i*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[N-1, i-1])/(2.0*dx)
                + dt*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
                ) # checked
            p_now[i, N-1]=(
                p_last[i, N-1]
                + dt*(force1(i*dx+dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[i+1, N-1] - force1(i*dx-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay, H)*p_last[i-1, N-1])/(2.0*dx)
                + dt*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(i*dx, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[i, 0] - force2(i*dx, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay, A)*p_last[i, N-2])/(2.0*dx)
                + dt*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
                ) # checked

        E_after_relax = 0.0 #intialize energy after relax, ie evolving probability distribution

        for i in range(N):
            for j in range(N):
                E_after_relax += potential(i*dx, j*dx, Ax, Axy, Ay)*p_now[i, j]

        heat += E_after_relax - E_change_pot #adds to cumulative heat
        heat_inst = E_after_relax - E_change_pot #heat just in this move

        calc_flux(p_now, flux, period, M_tot, m1, m2, gamma, beta, Ax, Axy, Ay, H, A, N, dx, dt)

        t += dt

    return work/cycles, heat/cycles