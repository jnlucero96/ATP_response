#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 7 September 2018 15:33:24

# Import internal python libraries

# Import external python libraries
from numpy import (
    array, zeros, einsum, where, finfo, float32, errstate, 
    abs as nabs, sum as nsum, nan as NaN
    )
from numpy.linalg import eig

# Import self-made libraries
from misc import safe_log

###############################################################################
################################### MATRICES ##################################
###############################################################################
def get_relax_matrix(a_val, b_val, c_val, d_val, *__):
    """\
    Description:

    Inputs:

    Outputs:
    """

    relax_matrix = array(
        [[1-a_val, b_val],
         [a_val, 1-b_val]],
        [[1-c_val, d_val],
         [c_val, 1-d_val]]
    )

    return relax_matrix


###############################################################################
############################## DISTRIBUTIONS ##################################
###############################################################################
def get_initial_distribution(
    initial_prob_x, relax_matrix, initialization_condition
    ):
    """\
    Description:

    Inputs:

    Outputs:
    """

    initial_prob = zeros((2, 2))
    relax_matrix_eq = einsum('i,ijk', initial_prob, relax_matrix)
    D, U = eig(relax_matrix_eq)

    if initialization_condition.lower() == 'equilibrium':
        # find the eigenvector corresponding to eigenvalue 1
        print("Initializing from: " + initialization_condition)
        p = array(U[:, where(nabs(D - 1.0) <= finfo(float32).eps)[0][0]])
        # normalize the distribution
        p /= p.sum()
    elif initialization_condition.lower() == 'steady state':
        #TODO: Implement this condition
        print("Initializing from: " + initialization_condition)
        p = array([0.5, 0.5])
    else:
        print(
            "Initialization condition not understood. \
            Starting from uniform distribuiton."
        )
        p = array([0.5, 0.5])

    p_initial = einsum('i,j->ij', initial_prob_x, p)

    assert abs(p_initial.sum() - 1.0) <= finfo(float32).eps, \
        "Initial distribution not normalized!"

    return p_initial

###############################################################################
############################### PROBABILITIES #################################
###############################################################################
def marginal_probability_X(p_XY, keepdims=True):
    """\
    Description: Calculate the marginal probability p[X_t]

    Inputs:

    Outputs:
    """
    return p_XY.sum(axis=-1, keepdims=keepdims)

def marginal_probability_Y(p_XY, keepdims=True):
    """\
    Description: Calculate the marginal probability p[Y_t]

    Inputs:

    Outputs:
    """
    return p_XY.sum(axis=-2, keepdims=keepdims)


def conditional_probability_X(p_XY):
    """\
    Description: Calculate the conditional probability p[X_t|Y_t].

    Inputs:

    Outputs:
    """
    with errstate(divide='ignore', invalid='ignore'):
        p_Y = marginal_probability_Y(p_XY)
        p_XgivenY = p_XY / p_Y
        p_XgivenY[where(nabs(p_Y) <= finfo(float32).eps)] = 0.0

    return p_XgivenY

def conditional_probability_Y(p_XY):
    """\
    Description: Calculate the conditional probability p[Y_t|X_t].

    Inputs:

    Outputs:
    """
    with errstate(divide='ignore', invalid='ignore'):
        p_X = marginal_probability_X(p_XY)
        p_YgivenX = p_XY / p_X
        p_YgivenX[where(nabs(p_X) <= finfo(float32).eps)] = 0.0
    return p_YgivenX



###############################################################################
############################## ENTROPIES ######################################
###############################################################################
def joint_entropy(p_XY, log_base):
    """\
    Description: Calculate the joint environment-system entropy H[X_t, Y_t] \
    from the joint distribution p[X_t, Y_t]

    Inputs:

    Outputs:
    """
    return nsum(p_XY*safe_log(p_XY, log_base), axis=(-1, -2)).__neg__()

def marginal_entropy_X(p_XY, log_base):
    """\
    Description: Calculate the environment marginal entropy H[X_t] from the \
    joint distribution p[X_t, Y_t]. 

    Inputs:

    Outputs:
    """
    p_X = marginal_probability_X(p_XY)
    return nsum(p_X*safe_log(p_X, log_base), axis=(-1, -2)).__neg__()

def marginal_entropy_Y(p_XY, log_base):
    """\
    Description: Calculate the system marginal entropy H[Y_t] from the \
    joint distribution p[X_t, Y_t].

    Inputs:

    Outputs:
    """
    p_Y = marginal_probability_Y(p_XY)
    return nsum(p_Y*safe_log(p_Y, log_base), axis=(-1, -2)).__neg__()
    
def conditional_entropy_X(p_XY, log_base):
    """\
    Description: Calculate the conditional entropy of the environment given \
    system (ie. H[X_t|Y_t])
    

    Inputs:

    Outputs:
    """
    p_XgivenY = conditional_probability_X(p_XY)
    H_XgivenY = nsum(
        p_XY*safe_log(p_XgivenY, log_base), axis=(-1, -2)
    ).__neg__()

    return H_XgivenY

def conditional_entropy_Y(p_XY, log_base):
    """\
    Description: Calculate the conditional entropy of the system given \
    environment (ie. H[Y_t|X_t])

    Inputs:

    Outputs:
    """
    p_YgivenX = conditional_probability_X(p_XY)
    H_YgivenX = nsum(
        p_XY*safe_log(p_YgivenX, log_base), axis=(-1, -2)
    ).__neg__()

    return H_YgivenX

def transfer_entropy(p_XY_t, log_base):
    """\
    Description: Calculate the transfer entropy

    Inputs:

    Outputs:
    """
    p_XY_t_minus_1 = nsum(p_XY_t, axis=(0, -1), keepdims=True)
    p_Y_t_minus_1 = nsum(p_XY_t, axis=0, keepdims=True)
    p_X_t_minus_1 = nsum(p_XY_t, axis=-1, keepdims=True)
    
    return p_XY_t * (
        safe_log(p_XY_t*p_XY_t_minus_1, log_base) 
        - safe_log(p_Y_t_minus_1*p_X_t_minus_1, log_base)
        ).sum()

def sigma_environment(p_XY, work_matrix, log_base):
    """\
    Description: Calculate the thermodyanmics entropy produced by the \
    environment.

    Inputs:

    Outputs:
    """
    p_X = marginal_probability_X(p_XY, keepdims=False)
    return (
        einsum('...i,...ij->...ij',p_X, work_matrix) 
        * (safe_log(work_matrix, log_base) - safe_log(work_matrix.T, log_base))
        ).sum()

def sigma_system(p_XY, relax_matrix, log_base):
    """\
    Description: Calculate the thermodynamic entropy produced by the system.

    Inputs:

    Outputs:
    """
    return (
        einsum('...i,...ij->...ij', relax_matrix, p_XY)
        * (
            safe_log(relax_matrix, log_base) 
            - safe_log(einsum('ijk->ikj',relax_matrix), log_base)
            )
    ).sum()


###############################################################################
############################## QUANTITIES #####################################
###############################################################################
def get_quantities(
    var_list, t_max, n_grid, log_base, pars, quantities, relax_distributions, 
    work_distributions, work_matrix, relax_matrix, par_tuple, t, param_func, 
    instantaneous
    ):
    """\
    Description:

    Inputs:

    Outputs:
    """

    if not instantaneous:
        t = 0

    index = (t,) + par_tuple

    try:
        quantities['H[Y_t]'][index] = \
            marginal_entropy_Y(relax_distributions, log_base)
    except KeyError:
        for var in var_list:
            if not instantaneous:
                quantities[var] = zeros(
                    (1,) + (n_grid,)*len(par_tuple)
                )
            else:
                quantities[var] = zeros(
                    (t_max + 2,) + (n_grid,)*len(par_tuple)
                )
            return get_quantities(
                var_list, t_max, n_grid, log_base, quantities, 
                relax_distributions, work_distributions, work_matrix, 
                relax_matrix, par_tuple, t, param_func, instantaneous
            )

    quantities['H[X_t]'][index] = \
        marginal_entropy_X(relax_distributions, log_base)

    quantities['H[Y_t|X_t]'][index] = \
        conditional_entropy_Y(relax_distributions, log_base)
    quantities['H[X_t|Y_t]'][index] = \
        conditional_entropy_X(relax_distributions, log_base)

    quantities['H[X_t,Y_t]'][index] = \
        quantities['H[Y_t]'][index] + quantities['H[X_t|Y_t]'][index]
    quantities['H[X_t+1,Y_t]'][index] = \
        quantities['H[Y_t]'][index] + quantities['H[X_t+1|Y_t]']
    
    quantities['I[X_t,Y_t]'][index] = \
        quantities['H[Y_t]'][index] - quantities['H[Y_t|X_t]'][index]
    quantities['I[X_t+1,Y_t]'][index] = \
        quantities['H[Y_t]'][index] - quantities['H[Y_t|X_t+1]'][index]
    
    with errstate(divide='ignore', invalid='ignore'):
        if nabs(quantities['I[X_t,Y_t]'][index]) >= finfo(float32).eps:
            quantities['r(t)'][index] = \
                quantities['I[X_t+1,Y_t'][index] \
                / quantities['I[X_t, Y_t]'][index]
        else: 
            quantities['r(t)'][index] = NaN

    quantities['sigma_X(t)'][index] = \
        sigma_environment(relax_distributions, work_matrix, log_base)
    quantities['sigma_Y(t)'][index] = \
        sigma_system(work_distributions, relax_matrix, log_base)
    quantities['sigma(t)'][index] = \
        quantities['sigma_X(t)'][index] + quantities['sigma_Y(t)'][index]

    quantities['F'][index] = F_neq_add_expectation(
        work_distributions, pars, param_func
    )
    

    
    

    
