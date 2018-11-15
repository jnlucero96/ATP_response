#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdarg>
#include <vector>
using namespace std;

double calc_flux(double);
double mean(double *);
void save_data(double *);
void save_flux_data(double *);
void run_calculation(
    double [], long, double, double, double, double, 
    double, double, double, double, double *
    );

ofstream ofile;

int main(int argc, char *argv[]) {
    
    // declare variables and associated values
    string outputfilename = "output.dat"; // output file name
    string outfluxfilename = "flux.dat"; // output file name
    double output_list[4]; // list to keep outputs in
    int N = 100; // declare discretization of space
    int cycles = 10; // maximum number of cycles
    int period = 0;
    double m = 1.0; // mass
    double gamma = 1000.0; // drag coefficient
    double A = 0.0; // amplitude of perturbing potential
    double k = 40.0; // tightness of curvature
    double dt = 0.1; // how big time step
    double beta = 1.0; // thermodynamic beta
    double E_i; // energy of ith state
    double dx = (2*M_PI) / ((double) N);
    double time_check = (dx * m * gamma)/ (1.5 * A + 0.5 * k);

    double work;
    double heat; 
    double flux[N];

    if (dt > time_check) {
        cout << "Time Unstable. Will not start calculation." << endl;
    } // end of if statement

    // make list of positions, length N, from 0 to 2Pi-dx
    double position[N];
    double p_init[N];
    double p_now[N];
    for (int i=0; i<N; i++) {
        position[i] = ((double) i) * ((2*M_PI) / (double) N);
    }

    // initialize and calculate partition function
    double Z = 0.0;
    for (int i=0; i<N; i++) {
        Z += exp(-beta*potential(0.0, position[i])) * dx;
    }

    // initialize and calculate the energies of the system
    double E_initial = 0.0;

    for (int i=0; i<N; i++) {
        E_i = potential(0.0, position[i]);
        p_init[i] += exp(-beta*E_i) * dx / Z;
        E_initial += E_i;
    } // end of for loop

    copy(p_init+0, p_init + N, p_now); // copy the array

    double E_0 = E_initial; // keep the initial energy

    delete[] p_init;

    run_calculation(
        p_now, N, dx, dt, E_initial, 
        ((double) cycles), ((double) period), m, gamma, beta, 
        work, heat, flux
        );

    ofile.open(outputfilename);

    save_flux_data(N, position, flux);

    return 0;
}

void run_calculation(
    double p_now[], long N, double dx, double dt, double E_now, 
    double cycles, double period, double m, double gamma, double beta,
    double work, double heat, double *flux) 
{

    long last = N - 1;         // variable to get last entry of array
    double E_change_potential; // energy of system after potential moves
    double E_after_relax; // energy of system after relaxation step
    // double work;
    double work_inst;
    // double heat;
    double heat_inst;
    // initialize flux array
    // double flux[N];
    double flux_now[N]; 
    double p_last[N];
    for (int i=0; i<N; i++) {
        flux[i] = 0.0;
    }

    // initialize some tracking variables
    double work;
    double heat;
    double t;
    work = 0;
    heat = 0;
    t = 0;
    long step_counter = 0;
    long print_counter = 0;

    t += dt;

    while (t < (cycles*period+dt)) {
        copy(p_now+0, p_now+N, p_last); // save previous distribution
        // reset current distribution and instantaneous flux
        for (int i=0; i<N; i++) {
            p_now[i] = 0.0;
            flux_now[i] = 0.0;
        } // end of for loop
        
        E_change_potential = 0.0;
        for (int i=0; i<N; i++) {
            E_change_potential += potential(t, ((double) i)*dx) * p_last;
        } // end of for loop
        
        work += (E_change_potential - E_now);
        work_inst = E_change_potential - E_now;
        
        // periodic boundary conditions; do first and last point explicitly
        p_now[0] = (
            p_last 
            + dt*(-force(t, dx)*p_last[1] + force(t, -dx)*p_last[N-1])/(2.0*dx+gamma*m)
            + dt*(p_last[1]+p_last[N-1]-2*p_last[0])/ pow(beta*gamma*dx, 2.0)
            );

        flux[0] += calc_flux(0);
        flux_now[0] = calc_flux(0);

        p_now[last] = (
            p_last[last]
            + dt*(
                -force(t, 0)*p_last[0] 
                + force(t, (((double) N-1)-1)* dx) 
                )/(2.0*dx+gamma*m)
            + dt*(p_last[0]+p_last[last-1]-2*p_last[-1])/pow(beta*gamma*dx, 2.0)
            );

        flux[last] += (
            force(t, ((((double) N)-1)*dx))*p_now[-1]/(m*gamma)
            - (-p_now[0]-p_now[N-2])/(gamma*beta*2*dx)
        )*dt/dx;
        flux_now[last] = (
            force(t, ((((double) N) - 1)*dx)) * p_now[last]/(m*gamma)
            - (p_now[0]-p_now[-2])/(gamma*beta*2*dx)
            )*dt/dx;

        for (int i=1; i<last; i++) {
            p_now[i] = (
                p_last[i]
                + dt*(
                    -force(t, (((double) i)+1)*dx)*p_last[i+1]
                    +force(t,(((double) i)-1)*dx)*p_last[i-1]
                    )/(2.0*dx*gamma*m)
                + dt*(p_last[i+1]+p_last[i-1]-2*p_last[i])/pow(beta*gamma*dx, 2.0)
            );
            flux[i] += calc_flux(i);
            flux_now[i] = calc_flux(i);
        } // end of for loop

        E_after_relax = 0.0;
        for (int i=0; i<N; i++) {
            E_after_relax += potential(t, ((double) i)*dx)*p_now[i];
        }// end of for loop

        heat += (E_after_relax-E_change_potential); // add to cumulative heat
        heat_inst = (E_after_relax - E_change_potential);

        t += dt;
    } // end of while loop

}

double mean(int count, double *array) {
    double mean_val = 0.0;

    for (int i=0; i<count; i++) {
        mean_val += array[i];
    } // end of for loop

    mean_val /= ((double) count);

    return mean_val;
}

void save_data(int count, ...) {

    ofile << setiosflags(ios::showpoint|ios::scientific);

    va_list args;
    va_start(args, count);

    for (int i=0; i<count; i++) {
        ofile << setw(15) << setprecision(20) << va_arg(args, double);
    } // end of for loop
    va_end(args);

    ofile << endl;

}

void save_flux_data(int count, double *theta0_array, double *flux_array) {

    ofile << setiosflags(ios::showpoint|ios::scientific);

    for (int i=0; i<count; i++) {
        ofile << setw(15) << setprecision(20) << theta0_array[i] << flux_array[i] << endl;
    } // end of for loop

    ofile << endl;

}

