#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 7 September 2018 15:33:24

# Import internal python libraries

# Import external python libraries
from numpy import (
    array, zeros, einsum, where, finfo, float32, float64, errstate, dot,
    abs as nabs, sum as nsum, nan as NaN
    )
from numpy.linalg import eig

# Import self-made libraries
from misc import safe_log
from dynamics import relax_step

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
        [
            [[1-a_val, b_val],
             [a_val,   1-b_val]],
            [[1-c_val, d_val],
             [c_val,   1-d_val]]
             ]
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

    relax_matrix_eq = einsum('i,ijk', initial_prob_x, relax_matrix)
    D, U = eig(relax_matrix_eq)

    if initialization_condition.lower() == 'equilibrium':
        # find the eigenvector corresponding to eigenvalue 1
        p = array(U[:, where(nabs(D - 1.0) <= finfo(float32).eps)[0][0]])
        # normalize the distribution
        p /= p.sum()
    elif initialization_condition.lower() == 'steady state':
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

def get_equilibrium_distribution(
    a_val, b_val, c_val, d_val, e_val, f_val
    ):
    """\
    Description:

    Inputs:

    Outputs:
    """

    work_matrix = array(
        [[1-e_val, 0,       f_val,   0      ],
         [0,       1-e_val, 0,       f_val  ],
         [e_val,   0,       1-f_val, 0      ],
         [0,       e_val,   0,       1-f_val]]
        )
    
    relax_matrix = array(
        [[1-a_val, b_val,   0,        0     ],
         [a_val,   1-b_val, 0,        0     ],
         [0,       0,       1-c_val, d_val  ],
         [0,       0,       c_val,   1-d_val]]
    )

    transfer_matrix = dot(relax_matrix, work_matrix)
    print(transfer_matrix)
    print(transfer_matrix[:,0].sum())
    D, U = eig(transfer_matrix)
    print(D);exit(0)
    # find vector corresponding with eigenvalue 1
    pi = array(U[:, where(nabs(D - 1.0) <= finfo(float32).eps)[0][0]])

    # normalize the vector
    pi /= pi.sum()

    assert abs(pi.sum() - 1.0) <= finfo(float32).eps, \
        "Equilibrium distribution not normalized!"

    return pi.reshape((2,2), order='C').real
    

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
    Description: Calculate the conditional probability p[X_t|Y_t] by \
    calculating the marginal distribution of X and then dividing it out from \
    the joint distribution. 

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

def relative_entropy(p,q, log_base):
    """\
    Description:

    Inputs:

    Outputs:
    """
    with errstate(divide='ignore'):
        return nsum(p*safe_log(p/q, log_base), axis=-1)

###############################################################################
############################# FREE ENERGY #####################################
###############################################################################

def F_neq_add(p_XY, pars, param_func, log_base):
    """\
    Description:

    Inputs:

    Outputs:
    """
    p_YgivenX = conditional_probability_Y(p_XY)
    a_val, b_val, c_val, d_val, *__ = param_func(pars)
    p_YgivenX_eq = array(
        [[b_val/(a_val+b_val), a_val/(a_val+b_val)],
         [d_val/(c_val+d_val), c_val/(c_val+d_val)]]
    )

    if a_val*b_val*c_val*d_val == 0:
        return array([0, 0])

    return relative_entropy(p_YgivenX, p_YgivenX_eq, log_base)

def F_neq_add_expectation(p_XY, pars, param_func, log_base):
    """\
    Description:

    Inputs:

    Outputs:
    """
    p_X = marginal_probability_X(p_XY, keepdims=False)
    return dot(p_X, F_neq_add(p_XY, pars, param_func, log_base))

def delta_F_neq_add_expectation(p_XY, pars, param_func, log_base):
    """\
    Description:

    Inputs:

    Outputs:
    """
    relax_matrix = get_relax_matrix(*param_func(pars))
    p_XY_t_plus_1 = relax_step(p_XY, relax_matrix)
    F_neq_add_now = F_neq_add(p_XY, pars, param_func, log_base)
    F_neq_add_after = F_neq_add(p_XY_t_plus_1, pars, param_func, log_base)
    delta_F_neq_add = F_neq_add_after - F_neq_add_now

    p_X = marginal_probability_X(p_XY, keepdims=False)
    return dot(p_X, delta_F_neq_add)

###############################################################################
############################## QUANTITIES #####################################
###############################################################################
def get_quantities(
    var_list, t_max, n_grid, log_base, pars, quantities, relax_distributions, 
    work_distributions, work_matrix, relax_matrix, par_tuple, t, param_func, 
    instantaneous
    ):
    """\
    Description: Get the quantities of interest from the simulation data.

    Inputs:

    Outputs:
    """

    if not instantaneous:
        t = 0

    index0 = (t,) + par_tuple
    index1 = (t-1,) + par_tuple

    try:
        quantities['H[Y_t]'][index0] = \
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
            var_list, t_max, n_grid, log_base, pars, quantities, 
            relax_distributions, work_distributions, work_matrix, 
            relax_matrix, par_tuple, t, param_func, instantaneous
        )
    
    quantities['H[X_t]'][index0] = \
        marginal_entropy_X(relax_distributions, log_base)

    quantities['H[Y_t|X_t]'][index0] = \
        conditional_entropy_Y(relax_distributions, log_base)
    quantities['H[X_t|Y_t]'][index0] = \
        conditional_entropy_X(relax_distributions, log_base)

    quantities['H[X_t,Y_t]'][index0] = \
        quantities['H[Y_t]'][index0] + quantities['H[X_t|Y_t]'][index0]
    quantities['H[X_t+1,Y_t]'][index0] = \
        quantities['H[Y_t]'][index0] + quantities['H[X_t+1|Y_t]'][index0]
    
    quantities['I[X_t,Y_t]'][index0] = \
        quantities['H[Y_t]'][index0] - quantities['H[Y_t|X_t]'][index0]
    quantities['I[X_t+1,Y_t]'][index0] = \
        quantities['H[Y_t]'][index0] - quantities['H[Y_t|X_t+1]'][index0]
    
    with errstate(divide='ignore', invalid='ignore'):
        if nabs(quantities['I[X_t,Y_t]'][index0]) >= finfo(float64).eps:
            quantities['r(t)'][index0] = \
                quantities['I[X_t+1,Y_t]'][index0] \
                / quantities['I[X_t,Y_t]'][index0]
        else: 
            quantities['r(t)'][index0] = NaN

    quantities['sigma_X(t)'][index0] = \
        sigma_environment(relax_distributions, work_matrix, log_base)
    quantities['sigma_Y(t)'][index0] = \
        sigma_system(work_distributions, relax_matrix, log_base)
    quantities['sigma(t)'][index0] = \
        quantities['sigma_X(t)'][index0] + quantities['sigma_Y(t)'][index0]

    quantities['F'][index0] = F_neq_add_expectation(
        work_distributions, pars, param_func, log_base
    )
    quantities['Delta F'][index0] = delta_F_neq_add_expectation(
        work_distributions, pars, param_func, log_base
    )
    quantities['-Delta F'][index0] = -quantities['Delta F'][index0]

    if t > 0 or not instantaneous:
        quantities['l_Y(t)'][index0] = \
            quantities['H[X_t+1|Y_t]'][index1] - quantities['H[X_t|Y_t]'][index0]
        quantities['I_nos(t)'][index1] = \
            quantities['I[X_t,Y_t]'][index1] - quantities['I[X_t+1,Y_t]'][index1]
        quantities['Delta I'][index1] = \
            quantities['I[X_t,Y_t]'][index0] - quantities['I[X_t,Y_t]'][index1]

        quantities['Delta H[X]'][index1] = \
            quantities['H[X_t]'][index0] - quantities['H[X_t]'][index1]
        quantities['Delta H[Y]'][index1] = \
            quantities['H[Y_t]'][index0] - quantities['H[Y_t]'][index1]
    
        
        quantities['I_e'][index1] = \
            quantities['H[Y_t|X_t]'][index1] - quantities['H[Y_t|X_t]'][index0]
        quantities['I_e^relax'][index1] = \
            quantities['H[Y_t|X_t+1]'][index1] - quantities['H[Y_t|X_t]'][index0]
        quantities['I_e^work'][index1] = \
            quantities['H[Y_t|X_t]'][index1] - quantities['H[Y_t|X_t+1]'][index1]
        
        quantities['W_diss(t)'][index1] = \
            quantities['I_nos(t)'][index1] + quantities['-Delta F'][index1]

        with errstate(divide='ignore', invalid='ignore'):
            if nabs(quantities['sigma_Y(t)'][index1]) > finfo(float64).eps:
                quantities['eta(t)'][index1] = \
                    quantities['l_Y(t)'][index1] \
                    / quantities['sigma_Y(t)'][index1]
            else:
                quantities['eta(t)'][index1] = NaN
            
            if nabs(quantities['W_diss(t)'][index1]) > finfo(float64).eps:
                quantities['mu(t)'][index1] = \
                    quantities['I_nos(t)'][index1] \
                    / quantities['W_diss(t)'][index1]
            else:
                quantities['mu(t)'][index1] = NaN

    return quantities

    
    

    
