# ATP_response
This is a repository containing all of the code which generates results for the manuscript titled "Nonequilibrium Response of Stochastic Strongly-Coupled Rotary Motors." by E. Lathouwers , J.N.E. Lucero, and D.A. Sivak

## Directories reference:
fokker_planck/working_directory_cython: **PRIMARY** directory containing the scripts that implement the integration of the Smoluchowski equation (using FTCS) and the associated analysis discussed in the manuscript.  

fokker_planck/working_directory_fortran: directory containing scripts which implement an alternate method to integrate the Smoluchowski equation using a first-order IMEX, pseudospectral method. 

fokker_planck/slurm_utilities: reference scripts used internally to parallelize submission of calculations on the ComputeCanada clusters.
