# cython: language_level=3
from libc.math cimport exp, sin, cos, fabs
import numpy as np
cimport numpy as np
from cython import cdivision, boundscheck
from sys import stdout, exit

DTYPE = np.longdouble
ctypedef np.longdouble_t DTYPE_t
cdef double pi = 3.141592653589793238462643383279502884197169399375105820974944592
cdef double thresh = 0.00000011920928955078 / 100.0 # single float precision

def launchpad(
    int steady_state, double cycles, int N, double write_out, 
    double period, double A, double k,
    double dt, double m, double beta, double gamma
    ):

    cdef:
        double dx         = (2*pi)/ N
        double time_check = dx*m*gamma / (1.5*A + 0.5*k)
        double Z          = 0.0
        double E          = 0.0
        double work          
        double heat
        double mean_flux
        double p_sum
        int    i  # declare general iterator variable
        int    x1  # declare position iterator
        int    x2  # declare position iterator
        double E_after_relax
        double E_0 
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] prob      = np.empty(N, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] p_now     = np.zeros(N, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] p_later   = np.empty(N, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] flux      = np.zeros(N, dtype=DTYPE)   # initalize cumulative flux as list to keep track of flux at every position
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] positions = np.linspace(0, (2*pi)-(2*pi/N), N, dtype=DTYPE)
    
    if dt > time_check:
        print("!!!TIME UNSTABLE!!!\n")

    for x1 in range(N):
        Z += exp(-beta*potential(0.0, positions[x1], period, A, k)) * dx
    
    for x2 in range(N):
        prob[x2] = exp(-beta*potential(0.0, positions[x2], period, A, k))*dx/Z
        E += potential(0.0, x2, period, A, k) * exp(-beta * potential(0.0, x2, period, A, k))*dx/Z
    
    # print(prob.tolist())
    if steady_state:
        print("Initializing from steady state...")
        init_simulation(prob, p_now, N, beta, gamma, dx, dt, m, period, A, k)
    else:
        print("Initializing from Gibbs-Boltzmann Distribution...")
        p_now = prob
    # print(p_now.tolist())
    # exit(0)
    
    E_after_relax = E
    E_0 = E

    print(p_now)
    print("Running simulation now")
    work, heat, mean_flux, p_sum = run_simulation_driven(
        p_now, flux, dx, dt, cycles, period, m, gamma, beta, A, k, E_after_relax
    )
    print(p_now)
    exit(0)

    return flux, mean_flux, work, heat, p_sum, p_now, prob

@cdivision(True)
cdef double trap_min(double t, double period):  #define dynamics of the potential minimum. Code only calls force and potential
    cdef double mini = 2*pi*t/period  #replace 2pi with the width of your space 
    while mini > 2*pi: #reset to 0 after each period
        mini -= 2*pi
    return mini

@cdivision(True)
cdef double force(
    double t, double position, double period, 
    double A, double k
    ): #need the force to calculate work
    cdef double x0 = trap_min(t, period)
    cdef double f = -1.5*A*cos(3*(position+pi/2))-0.5*k*sin(position-x0)
    # f=-k*(position-pi)
    return f

@cdivision(True)
cdef double potential(
    double t, double position, double period, double A, double k
    ): #need the potential in the FPE
    cdef double x0 = trap_min(t, period)
    cdef double E  = A*(1+sin(3.0*(position+pi/2.0)))/2.0 + k*(1+sin(position-(pi/2)-x0))/2
    # E=0.5*k*(position-pi)**2
    return E

@cdivision(True)
cdef double calculate_flux(
    int location, double t, 
    np.ndarray[DTYPE_t, ndim=1, negative_indices=True, mode='c'] p_last, 
    double m, double gamma, double beta, double period, double dx, double dt, 
    double A, double k
    ):
     
    cdef double flux = (
            force(t, location*dx, period, A, k)*p_last[location]/(m*gamma)
            -(p_last[location+1]-p_last[location-1])/(gamma*beta*2*dx)
        )*dt/dx

    return flux

@cdivision(True)
@boundscheck(False)
cdef double calc_mean(
    np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] array, int N
    ):
    cdef double mean_val = 0.0
    cdef int i 
    for i in range(N):
        mean_val += array[i]

    return mean_val / N

@boundscheck(False)
cdef double calc_sum(
    np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] array, int N
    ):
    cdef double sum_val = 0.0
    cdef int i 

    for i in range(N):
        sum_val += array[i]

    return sum_val

@cdivision(True)
@boundscheck(False)
cdef init_simulation(
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] p_equil, 
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] return_array, 
        int N, double beta, double gamma, double dx, 
        double dt, double m, double period, double A, 
        double k
    ):

    cdef: 
        int i
        int condition = 1
        int iteration = 0
        double t = 0.0
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] p_now  = np.empty(N, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] p_last = np.empty(N, dtype=DTYPE)

    for i in range(N):
        p_now[i] = p_equil[i]
    
    while True:

        condition = 1
        t += dt
        while t < (period+dt): # run protocol

            for i in range(N):
                p_last[i] = p_now[i] # save previous distribution
                p_now[i] = 0.0 # reset to zero

            p_now[0] = (
                p_last[0] + dt*(-force(t, dx, period, A, k)*p_last[1] 
                + force(t, -dx, period, A, k)*p_last[N-1])/(2*dx*gamma*m)
                + dt*(p_last[1]+p_last[N-1]-2*p_last[0])/(beta*gamma*dx**2)
                )

            for i in range(1, N-1):
                p_now[i]=(
                    p_last[i] + dt*(-force(t, i*dx+dx, period, A, k)*p_last[i+1]
                    + force(t, i*dx-dx, period, A, k)*p_last[i-1])/(2*dx*gamma*m)
                    + dt*(p_last[i+1]+p_last[i-1]-2*p_last[i])/(beta*gamma*dx**2))

            p_now[N-1] = (
                p_last[N-1] + dt*(-force(t, 0, period, A, k)*p_last[0]
                + force(t, (N-1)*dx-dx, period, A, k)*p_last[N-2])/(2*dx*gamma*m)
                +dt*(p_last[0]+p_last[N-2]-2*p_last[N-1])/(beta*gamma*dx**2)
                )

            t += dt
        
        for i in range(N):
            if 0.5*fabs(p_last[i] - p_now[i]) >= thresh: # Check if PSS has been reached (3.15 in AKasper thesis)
                condition = 0
                break
            else:
                continue

        if not condition:
        
            for i in range(N):
                return_array[i] += p_now[i]

            return return_array
        else:
            t = 0.0
            iteration += 1

@cdivision(True)
@boundscheck(False)
cdef (double, double, double, double) run_simulation_driven(
    np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] p_now, 
    np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] flux, 
    double dx, double dt, double cycles, double period, double m, double gamma, 
    double beta, double A, double k, double E_after_relax
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
        int         i
        int         N             = p_now.size
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] p_last   = np.empty(N, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] flux_now = np.empty(N, dtype=DTYPE)  
    
    t += dt #step forwards in t

    while t < (cycles*period+dt):
        
        for i in range(N):
            p_last[i] = p_now[i] # save previous distribution
            # reset to zero
            p_now[i] = 0.0 
            flux_now[i] = 0.0
        
        E_last = E_after_relax  #Energy at t-dt is E_last
        E_change_pot = 0.0 #intialize energy of system after potential moved

        for i in range(N):
            E_change_pot += potential(t, i*dx, period, A, k)*p_last[i]
        work += E_change_pot - E_last #add to cululative work
        work_inst = E_change_pot - E_last #work just in this move
        
        ## since I have periodic boundary conditions, I explicitly write out the FPE update for x=0
        p_now[0] = (
            p_last[0] + dt*(-force(t, dx, period, A, k)*p_last[1] 
            + force(t, -dx, period, A, k)*p_last[N-1])/(2*dx*gamma*m)
            + dt*(p_last[1]+p_last[N-1]-2*p_last[0])/(beta*gamma*dx**2)
            )
        flux[0] += calculate_flux(
                0, t, p_last, m, gamma, beta, period, dx, dt, A, k
                )
        flux_now[0] = calculate_flux(
                0, t, p_last, m, gamma, beta, period, dx, dt, A, k
                )
        
        ## all points with well defined neighbours go like so:
        for i in range(1, N-1):
            p_now[i]=(
                p_last[i] 
                + dt*(
                    -force(t, i*dx+dx, period, A, k)*p_last[i+1] 
                    + force(t, i*dx-dx, period, A, k)*p_last[i-1]
                    )/(2*dx*gamma*m)
                + dt*(p_last[i+1]+p_last[i-1]-2*p_last[i])/(beta*gamma*dx**2))
            flux[i] += calculate_flux(
                i, t, p_last, m, gamma, beta, period, dx, dt, A, k
                )
            flux_now[i] = calculate_flux(
                i, t, p_last, m, gamma, beta, period, dx, dt, A, k
                )
        
        # due to periodic BC, again do last point explicitly
        p_now[N-1] = (
            p_last[N-1] 
            + dt*(
                -force(t, 0, period, A, k)*p_last[0] 
                + force(t, (N-1)*dx-dx, period, A, k)*p_last[N-2]
                )/(2*dx*gamma*m)
            + dt*(p_last[0]+p_last[N-2]-2*p_last[N-1])/(beta*gamma*dx**2)
            )
        flux[N-1] += (
            force(t, (N-1)*dx, period, A, k)*p_now[N-1]/(m*gamma)
            - (p_now[0]-p_now[N-2])/(gamma*beta*2*dx)
            )*dt/dx
        flux_now[N-1] = (
            force(t, (N-1)*dx, period, A, k)*p_now[N-1]/(m*gamma)
            - (p_now[0]-p_now[N-2])/(gamma*beta*2*dx)
            )*dt/dx

        E_after_relax = 0.0 #intialize energy after relax, ie evolving probability distribution
        for i in range(N):
            E_after_relax += potential(t, i*dx, period, A, k)*p_now[i]
        
        heat += E_after_relax - E_change_pot #adds to cumulative heat
        heat_inst = E_after_relax - E_change_pot #heat just in this move

        t += dt

    return work, heat, calc_mean(flux, N), calc_sum(p_now, N)#, calc_sum(p_now, N)


