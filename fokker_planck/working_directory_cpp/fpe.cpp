#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdarg>
#include <vector>
#include "math.h"
using namespace std;

// ==================== DECLARE GLOBAL VARIABLES ============================
// yes, this is what you think it is
const double pi = 3.14159265358979323846264338327950288419716939937510582;
// float32 machine eps
const double float32_eps = 1.1920928955078125e-07;
// float64 machine eps
const double float64_eps = 2.22044604925031308084726e-16;

// ===========================================================================
// ===========
// =========== DECLARE PARAMETERS HERE
// ===========
// ===========================================================================

// discretization parameters
const double dt = 0.001; // time discretization. Keep this number low
const double check_step = ((int) 1.0/dt); // check step counter

const int N = 360; // inverse space discretization. Keep this number high!
const double dx = (2.0*M_PI)/N; // space discretization

//# model-specific parameters
const double gamma_var = 1000.0; // drag
const double beta_var = 1.0; // 1/kT
const double m1 = 1.0; // mass of system 1
const double m2 = 1.0; // mass of system 2

const double E0 = 3.0; // energy scale of F0 sub-system
const double Ecouple = 3.0; // energy scale of coupling between sub-systems F0 and F1
const double E1 = 3.0; // energy scale of F1 sub-system
const double F_Hplus = 3.0; // energy INTO (positive) F0 sub-system by H+ chemical bath
const double F_atp = 3.0; // energy INTO (positive) F1 sub-system by ATP chemical bath

const double num_minima = 3.0; // number of minima in the potential
const double phase_shift = 0.0; // how much sub-systems are offset from one another

// prototypes for functions
double force1(
    double, double
    );
double force2(
    double, double
    );
double potential(
    double, double
    );
void update_probability_full(
    double[][N], double[][N], double[][N], double[][N]
    );
void steady_state_initialize(
    double[][N], double[][N], double[][N], double[][N], double[][N]
    );
void launchpad_reference(
    double[N],
    double[][N], double[][N], double[][N], double[][N], double[][N],
    double[][N], double[][N]
    );
void linspace(double, double, double [N]);

ofstream ofile;

int main(int argc, char *argv[]) {

    double prob[N][N];
    double p_now[N][N];
    double p_last[N][N];
    double p_last_ref[N][N];
    double positions[N];
    double potential_at_pos[N][N];
    double force1_at_pos[N][N];
    double force2_at_pos[N][N];

    size_t i, j;

    // populate the positions array
    linspace(0.0, (2.0*M_PI)-dx, positions);

    // initialize all other arrays to zero
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            prob[i][j] = 0.0;
            p_now[i][j] = 0.0;
            p_last[i][j] = 0.0;
            p_last_ref[i][j] = 0.0;
            potential_at_pos[i][j] = 0.0;
            force1_at_pos[i][j] = 0.0;
            force2_at_pos[i][j] = 0.0;
        }
    }

    launchpad_reference(
        positions, prob, p_now, p_last, p_last_ref, 
        potential_at_pos, force1_at_pos, force2_at_pos
    );

    return 0;
}

void linspace(double a, double b, double array[N]) {

    double delta=(b-a)/(N-1);

    for (int i=0; i<N; i++) {
            array[i] = a + (i*delta);
    }
}

void launchpad_reference(
    double positions[N],
    double prob[][N], double p_now[][N],
    double p_last[][N], double p_last_ref[][N],
    double potential_at_pos[][N],
    double force1_at_pos[][N], double force2_at_pos[][N]
    ) {
    
    double Z = 0.0;

    size_t i, j; // declare iterator variables

    // populate the reference arrays
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            potential_at_pos[i][j] = potential(positions[i], positions[j]);
            force1_at_pos[i][j] = force1(positions[i], positions[j]);
            force2_at_pos[i][j] = force2(positions[i], positions[j]);
        }
    }

    // calculate the partition function
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            Z += exp(-beta_var*potential_at_pos[i][j]);
        }
    }

    // calculate the boltzmann equilibrium function and the average energy
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            prob[i][j] = exp((-1.0)*beta_var*potential_at_pos[i][j])/Z;
        }
    }

    // initialize the simulation to steady state distribution
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            p_now[i][j] = 1.0/(N*N);
        }
    }

    steady_state_initialize(
        p_now, p_last, p_last_ref,
        force1_at_pos, force2_at_pos
    );

    //TODO: add a write out to file here.
}

double force1( double position1, double position2) {
    return (0.5)*(Ecouple*sin(position1-position2)
        + (num_minima*E0*sin((num_minima*position1)-(phase_shift)))) 
        - F_Hplus;
}

double force2( double position1, double position2) {
    return (0.5)*((-1.0)*Ecouple*sin(position1-position2)
        + (num_minima*E1*sin(num_minima*position2))) 
        - F_atp;
}

double potential( double position1, double position2) {
    return 0.5*(
        E0*(1-cos((num_minima*position1-phase_shift)))
        + Ecouple*(1-cos(position1-position2)) 
        + E1*(1-cos((num_minima*position2)))
        );
}

void steady_state_initialize(
    double p_now[][N],
    double p_last[][N],
    double p_last_ref[][N],
    double force1_at_pos[][N],
    double force2_at_pos[][N]
    ) {

    double tot_var_dist = 0.0;
    int continue_condition = 1;
    double check_norm = 0.0;

    // counters
    size_t i, j;
    unsigned long step_counter = 0;

    while (continue_condition) {

        for (i=0; i<N; i++) {
            for (j=0; j<N; j++) {
                // save previous distribution
                p_last[i][j] = p_now[i][j];
                // reset to zero
                p_now[i][j] = 0.0;
            }
        }

        // advance probability one time step
        update_probability_full(
            p_now, p_last,
            force1_at_pos, force2_at_pos
            );

        if (step_counter == check_step) {
            for (i=0; i<N; i++) {
                for (j=0; j<N; j++) {
                    tot_var_dist += 0.5*fabs(p_last_ref[i][j] - p_now[i][j]);
                }
            }

            for (i=0; i<N; i++) {
                for (j=0; j<N; j++) {
                    check_norm += p_now[i][j];
                }
            }
            
            // bail at the first sign of trouble
            if (fabs(check_norm-1.0) <= float32_eps) {exit(1);}

            // check condition
            if (tot_var_dist < float64_eps) {
                continue_condition = 0;
            } else {
                tot_var_dist = 0.0; // reset total variation distance
                step_counter = 0; // reset step counter
                check_norm = 0.0;
                // make current distribution the reference distribution
                for (i=0; i<N; i++) {
                    for (j=0; j<N; j++) {
                        p_last_ref[i][j] = p_now[i][j];
                    }
                }
            }
        }

        step_counter += 1;
    }
}   

void update_probability_full(
    double p_now[][N],
    double p_last[][N],
    double force1_at_pos[][N],
    double force2_at_pos[][N]
    ) {

    // declare iterator variables
    int i, j;

    // Periodic boundary conditions:
    // Explicity update FPE for the corners
    p_now[0][0] = (
        p_last[0][0]
        + dt*(force1_at_pos[1][0]*p_last[1][0]-force1_at_pos[N-1][0]*p_last[N-1][0])/(gamma_var*m1*2.0*dx)
        + dt*(p_last[1][0]-2.0*p_last[0][0]+p_last[N-1][0])/(beta_var*gamma_var*m1*(dx*dx))
        + dt*(force2_at_pos[0][1]*p_last[0][1]-force2_at_pos[0][N-1]*p_last[0][N-1])/(gamma_var*m2*2.0*dx)
        + dt*(p_last[0][1]-2.0*p_last[0][0]+p_last[0][N-1])/(beta_var*gamma_var*m2*(dx*dx))
        ); // checked
    p_now[0][N-1] = (
        p_last[0][N-1]
        + dt*(force1_at_pos[1][N-1]*p_last[1][N-1]-force1_at_pos[N-1][N-1]*p_last[N-1][N-1])/(gamma_var*m1*2.0*dx)
        + dt*(p_last[1][N-1]-2.0*p_last[0][N-1]+p_last[N-1][N-1])/(beta_var*gamma_var*m1*(dx*dx))
        + dt*(force2_at_pos[0][0]*p_last[0][0]-force2_at_pos[0][N-2]*p_last[0][N-2])/(gamma_var*m2*2.0*dx)
        + dt*(p_last[0][0]-2.0*p_last[0][N-1]+p_last[0][N-2])/(beta_var*gamma_var*m2*(dx*dx))
        ); // checked
    p_now[N-1][0] = (
        p_last[N-1][0]
        + dt*(force1_at_pos[0][0]*p_last[0][0]-force1_at_pos[N-2][0]*p_last[N-2][0])/(gamma_var*m1*2.0*dx)
        + dt*(p_last[0][0]-2.0*p_last[N-1][0]+p_last[N-2][0])/(beta_var*gamma_var*m1*(dx*dx))
        + dt*(force2_at_pos[N-1][1]*p_last[N-1][1]-force2_at_pos[N-1][N-1]*p_last[N-1][N-1])/(gamma_var*m2*2.0*dx)
        + dt*(p_last[N-1][1]-2.0*p_last[N-1][0]+p_last[N-1][N-1])/(beta_var*gamma_var*m2*(dx*dx))
        ); // checked
    p_now[N-1][N-1] = (
        p_last[N-1][N-1]
        + dt*(force1_at_pos[0][N-1]*p_last[0][N-1]-force1_at_pos[N-2][N-1]*p_last[N-2][N-1])/(gamma_var*m1*2.0*dx)
        + dt*(p_last[0][N-1]-2.0*p_last[N-1][N-1]+p_last[N-2][N-1])/(beta_var*gamma_var*m1*(dx*dx))
        + dt*(force2_at_pos[N-1][0]*p_last[N-1][0]-force2_at_pos[N-1][N-2]*p_last[N-1][N-2])/(gamma_var*m2*2.0*dx)
        + dt*(p_last[N-1][0]-2.0*p_last[N-1][N-1]+p_last[N-1][N-2])/(beta_var*gamma_var*m2*(dx*dx))
        ); // checked

    // iterate through all the coordinates, not on the corners, for both variables
    for (i=1; i<N-1; i++) {
        // Periodic boundary conditions:
        // Explicitly update FPE for edges not corners
        p_now[0][i] = (
            p_last[0][i]
            + dt*(force1_at_pos[1][i]*p_last[1][i]-force1_at_pos[N-1][i]*p_last[N-1][i])/(gamma_var*m1*2.0*dx)
            + dt*(p_last[1][i]-2*p_last[0][i]+p_last[N-1][i])/(beta_var*gamma_var*m1*(dx*dx))
            + dt*(force2_at_pos[0][i+1]*p_last[0][i+1]-force2_at_pos[0][i-1]*p_last[0][i-1])/(gamma_var*m2*2.0*dx)
            + dt*(p_last[0][i+1]-2*p_last[0][i]+p_last[0][i-1])/(beta_var*gamma_var*m2*(dx*dx))
            ); // checked
        p_now[i][0] = (
            p_last[i][0]
            + dt*(force1_at_pos[i+1][0]*p_last[i+1][0]-force1_at_pos[i-1][0]*p_last[i-1][0])/(gamma_var*m1*2.0*dx)
            + dt*(p_last[i+1][0]-2*p_last[i][0]+p_last[i-1][0])/(beta_var*gamma_var*m1*(dx*dx))
            + dt*(force2_at_pos[i][1]*p_last[i][1]-force2_at_pos[i][N-1]*p_last[i][N-1])/(gamma_var*m2*2.0*dx)
            + dt*(p_last[i][1]-2*p_last[i][0]+p_last[i][N-1])/(beta_var*gamma_var*m2*(dx*dx))
            ); // checked

        // all points with well defined neighbours go like so:
        for (j=1; j<N-1; j++) { 
            p_now[i][j] = (
                p_last[i][j]
                + dt*(force1_at_pos[i+1][j]*p_last[i+1][j]-force1_at_pos[i-1][j]*p_last[i-1][j])/(gamma_var*m1*2.0*dx)
                + dt*(p_last[i+1][j]-2.0*p_last[i][j]+p_last[i-1][j])/(beta_var*gamma_var*m1*(dx*dx))
                + dt*(force2_at_pos[i][j+1]*p_last[i][j+1]-force2_at_pos[i][j-1]*p_last[i][j-1])/(gamma_var*m2*2.0*dx)
                + dt*(p_last[i][j+1]-2.0*p_last[i][j]+p_last[i][j-1])/(beta_var*gamma_var*m2*(dx*dx))
                ); // checked
        }

        // Explicitly update FPE for rest of edges not corners
        p_now[N-1][i] = (
            p_last[N-1][i]
            + dt*(force1_at_pos[0][i]*p_last[0][i]-force1_at_pos[N-2][i]*p_last[N-2][i])/(gamma_var*m1*2.0*dx)
            + dt*(p_last[0][i]-2.0*p_last[N-1][i]+p_last[N-2][i])/(beta_var*gamma_var*m1*(dx*dx))
            + dt*(force2_at_pos[N-1][i+1]*p_last[N-1][i+1]-force2_at_pos[N-1][i-1]*p_last[N-1][i-1])/(gamma_var*m2*2.0*dx)
            + dt*(p_last[N-1][i+1]-2.0*p_last[N-1][i]+p_last[N-1][i-1])/(beta_var*gamma_var*m2*(dx*dx))
            ); // checked
        p_now[i][N-1] = (
            p_last[i][N-1]
            + dt*(force1_at_pos[i+1][N-1]*p_last[i+1][N-1]-force1_at_pos[i-1][N-1]*p_last[i-1][N-1])/(gamma_var*m1*2.0*dx)
            + dt*(p_last[i+1][N-1]-2.0*p_last[i][N-1]+p_last[i-1][N-1])/(beta_var*gamma_var*m1*(dx*dx))
            + dt*(force2_at_pos[i][0]*p_last[i][0]-force2_at_pos[i][N-2]*p_last[i][N-2])/(gamma_var*m2*2.0*dx)
            + dt*(p_last[i][0]-2.0*p_last[i][N-1]+p_last[i][N-2])/(beta_var*gamma_var*m2*(dx*dx))
            ); // checked
    } // end of for loop
}

