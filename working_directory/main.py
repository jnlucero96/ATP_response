#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 7 September 2018 08:53:10

# Import internal python libraries
from sys import argv
from itertools import product

# Import external python libraries
from numpy import zeros, array, squeeze, nanmax, nanmin

# Import self-made libraries
from misc import get_vars, read_from_config, get_params
from parameterization import select_parameterization
from dynamics import work_step, relax_step
from quantities import (
    get_relax_matrix, get_initial_distribution, 
    get_quantities, get_equilibrium_distribution
    )
from analysis import plot_heatmap, plot_heatmap_grid, plot_timeseries

def main(argc, argv):

    """\
    Description:

    Inputs: 

    Outputs:
    """

    try:
        script_name, analyze, *args = argv
    except ValueError:
        print(
            "Not enough values to unpack. Assuming no analysis is to be done."
            )

    if analyze.lower() in ('y', 'yes', 't', 'true'):
        analyze = True
    elif analyze.lower() in ('n', 'no', 'f', 'false'):
        analyze = False

    var_names, var_units, var_list = get_vars()
    config_params = read_from_config()

    q_x = config_params.getfloat('q_x')
    gamma_x = config_params.getfloat('gamma_x')
    initial_prob_x = array(
        [config_params.getfloat('initial probability of system'),
         1-config_params.getfloat('initial probability of system')]
        )

    assert (
        set([config_params['display quantity']]).issubset(set(var_list))
        ), "Cannot display required quantities. Choose another set."

    print("Calculating desired distributions and quantities...")

    model_param_func, log_params, axes = select_parameterization(
        config_params['parametrization'], config_params.getfloat('q_x'),
        config_params.getfloat('gamma_x')
    )
    dimension = len(log_params)

    # Don't know what these quantities are
    e = gamma_x*q_x
    f = (1-q_x) * gamma_x

    # Don't really understand why the matrix is defined in terms of these
    # e and f quantities
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

    print("Initializing from: " + config_params['initialize from'])

    for par_tuple in par_tuple_generator:
        pars = get_params(
            par_tuple, config_params.getint('n_grid'), 
            log=log_params, base=config_params.getfloat('log base')
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
                config_params.getfloat('log base'), pars, quantities, 
                relax_distributions[(t,) + par_tuple],
                work_distributions[(t,) + par_tuple], work_matrix, relax_matrix,
                par_tuple, t, model_param_func, instantaneous=True
            )

        relax_distribution_eq = get_equilibrium_distribution(
            *model_param_func(pars)
            )
        work_distribution_eq = work_step(relax_distribution_eq, work_matrix)
        quantities = get_quantities(
            var_list, config_params.getint('t_max'),
            config_params.getint('n_grid'),
            config_params.getfloat('log base'), pars, quantities,
            relax_distributions[(t,) + par_tuple],
            work_distributions[(t,) + par_tuple], work_matrix, relax_matrix,
            par_tuple, t, model_param_func, instantaneous=False
        )
    
    for k in quantities.keys():
        quantities_eq[k] = squeeze(quantities_eq[k])

    print("Calculation Complete!")

    if analyze:
        print("Beginning analysis now...")

        if dimension == 4:

            for key, value in quantities.items():
                if not key in [config_params['display quantity']]:
                    continue    
                print("Plotting heatmap for " + key + "...")

                key_name = var_names[key]
                key_units = var_units[(key, config_params.getfloat('log base'))]

                if key_units == '':
                    maxval = 1
                    minval = 0
                else:
                    maxval = nanmax(value)
                    minval = min(nanmin(value), 0)

                for t_index in range(config_params.getint('t_max') + 1):
                    filename = key + '_grid_hmap_t_' + str(t_index) + '.pdf'
                    title = key_name + ' (t = ' + str(t_index) + ')'
                    plot_heatmap_grid(
                        value[t_index,...], key, key_name, key_units, axes, 
                        log_params,
                        title=title, filename=filename, 
                        maxval=maxval, minval=minval
                        )
                
                filename2 = key + '_grid_hmap_steady_state.pdf'
                title = key_name + ' (steady state)'
                if key_units == '':
                    maxval = 1
                    minval = 0
                else:
                    maxval = nanmax(quantities_eq[key])
                    minval = min(nanmin(quantities_eq[key]), 0)
                plot_heatmap_grid(
                    quantities_eq[key], key, key_name, key_units, axes,
                    log_params,
                    title=title, filename=filename,
                    maxval=maxval, minval=minval
                )
        elif dimension == 2:
            for key, value in quantities.items():
                if not key in [config_params['display quantity']]:
                    continue
                print("Plotting heatmap for " + key + "...")
                key_name = var_names[key]
                key_units = var_units[(key, config_params.getfloat('log base'))]

                if key_units == '':
                    maxval = 1
                    minval = 0
                else:
                    maxval = nanmax(value)
                    minval = min(nanmin(value), 0)
                
                for t_index in range(config_params.getint('t_max') + 1):
                    print("Plotting for t = " + str(t_index))
                    title = key_name + ' (t = ' + str(t_index) + ')'
                    filename = key + '_hmap_t_' + str(t_index) + '.pdf'
                    plot_heatmap(
                        value[t_index, ...], key, key_name, key_units, axes,
                        title=title, filename=filename,
                        maxval=maxval, minval=minval
                    )
                
                print("Plotting timeseries for " + key + "...")
                for index4 in range(config_params.getint('n_grid')):
                    print("Plotting for grid number: " + str(index4))
                    if log_params[1]:
                        title_timeseries = key_name + r' ($' + axes[1][1:-1] \
                            + r'=' + str(index4) + r'/' \
                            + str(config_params.getint('n_grid')-1) + r'$)'
                    else:
                        title_timeseries = key_name + r' ($' + axes[1][1:-1] \
                            + r'=' + str(index4) + r'/' \
                            + str(config_params.getint('n_grid')-1) + r'$)'
                    filename_timeseries = key + '_tseries' \
                        + axes[1][1:-1] + '_' + str(index4) + '.pdf'
                    plot_timeseries(
                        value[:-1, :, index4], key, key_name, key_units, 
                        config_params.getint('t_max'), axes, log_params, 
                        title=title_timeseries, filename=filename_timeseries, 
                        maxval=maxval, minval=minval
                        )

            print("Completed!")






    return 0

if __name__ == "__main__":
    main(len(argv), argv)
