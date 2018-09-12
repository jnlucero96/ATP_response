#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 7 September 2018 08:53:10

# Import internal python libraries
from itertools import product

# Import external python libraries
from numpy import zeros, array

# Import self-made libraries
from misc import get_vars, read_from_config, get_params
from parameterization import select_parameterization
from dynamics import work_step, relax_step
from quantities import (
    get_relax_matrix, get_initial_distribution, 
    get_quantities
    )

def main():

    """\
    Description:

    Inputs: 

    Outputs:
    """

    var_names, var_units, var_list = get_vars()
    config_params = read_from_config()

    q_x = config_params['q_x']
    gamma_x = config_params['gamma_x']
    initial_prob_x = array(
        [config_params['initial probability of system']]
        )

    assert(set(config_params['display_quantities'])).issubset(set(var_list)), \
        "Cannot display required quantities. Choose another set."

    print("Calculating desired distributions and quantities...")

    model_param_func, log_params, axes = select_parameterization(
        config_params['parametrization']
    )
    dimension = len(log_params)

    # Don't know what these quantities are
    e = gamma_x*q_x
    f = (1-q_x) * gamma_x

    work_matrix = array(
        [[1-e, f],
         [e, 1-f]]
    )

    # Initialize arrays and tables
    relax_distributions = work_distributions = zeros(
        (config_params.getint('t_max')+2,) 
        + (config_params.getint('n_grid'),)*dimension 
        + (2, 2)
        )
    par_array = zeros(
        (config_params.getint('n_grid'),)*dimension + (dimension,)
        )
    quantities = quantities_eq = {}

    par_tuple_generator = product(
        list(range(config_params.getint('n_grid'))), repeat=dimension
        )
        
    for par_tuple in par_tuple_generator:
        pars = get_params(
            par_tuple, config_params.getint('n_grid'), 
            log=log_params, base=config_params.getint('log base')
            )
        par_array[par_tuple,...] = pars

        relax_matrix = get_relax_matrix(*model_param_func(pars))
        p_XY = get_initial_distribution(
            initial_prob_x, relax_matrix, config_params['initialize from']
            )

        # Begin time series
        for t in range(0, config_params.getint('t_max')+2):
            relax_distributions[(t,) + par_tuple] = p_XY

            p_XY = work_step(p_XY, work_matrix)
            work_distributions[(t,) + par_tuple] = p_XY
            
            p_XY = relax_step(p_XY, relax_matrix)
            quantities = get_quantities(
                var_list, config_params.getint('t_max'), 
                config_params.getint('n_grid'), 
                config_params.getint('log base'), pars, quantities, 
                relax_distributions[(t,) + par_tuple],
                work_distributions[(t,) + par_tuple], work_matrix, relax_matrix,
                par_tuple, t, model_param_func, instantaneous=True
            )

    return 0

