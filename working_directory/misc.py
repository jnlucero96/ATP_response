#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 7 September 2018 09:20:50

# Import internal python libraries
from sys import exit
from os import getcwd

# Import external python libraries
from configparser import ConfigParser
from math import e as euler
from numpy import errstate, log as nlog
from numpy.ma import log

# Import self-made libraries

def get_vars():
    """\
    Description:
    Initialize into memory the variables of interest and get them ready as \
    plot labels. Also get units associated with the variables.

    Inputs:
    None

    Outputs:
    :param var_names: Map that turns variable into LaTeX code for plotting later
    :type var_names: dict (tuple: raw_str)
    :param var_units: Get units associated with the quantities
    :type var_units: dict (tuple: str)
    :param var_list: list form of all the variables
    :type var_list: list
    """

    var_names = {
        'H[Y_t]': r'$H[Y_{t}]$',
        'H[X_t]': r'$H[X_{t}]$',
        'H[Y_t|X_t]': r'$H[Y_{t}|X_{t}]$',
        'H[X_t|Y_t]': r'$H[X_{t}|Y_{t}]$',
        'H[Y_t|X_t+1]': r'$H[Y_{t}|X_{t+1}]$',
        'H[X_t+1|Y_t]': r'$H[X_{t+1}|Y_{t}]$',
        'H[X_t,Y_t]': r'$H[X_{t},Y_{t}]$',
        'H[X_t+1,Y_t]': r'$H[X_{t+1},Y_{t}]$',
        'I[X_t,Y_t]': r'$I[X_{t},Y_{t}]$',
        'I[X_t+1,Y_t]': r'$I[X_{t+1},Y_{t}]$',
        'sigma_X(t)': r'$\sigma_{X}(t)$',
        'sigma_Y(t)': r'$\sigma_{Y}(t)$',
        'sigma(t)': r'$\sigma(t)$',
        'l_Y(t)': r'$l_{Y}(t)$',
        'I_nos(t)': r'$I_{\mathrm{nos}}(t)$',
        'T_{X->Y}(t)': r'$T_{X \rightarrow Y}(t)$',
        'Delta I': r'$\Delta I[X,Y]$',
        'Delta H[X]': r'$\Delta H[X]$',
        'Delta H[Y]': r'$\Delta H[Y]$',
        'I_e': r'$I_{\mathrm{e}}$',
        'I_e^relax': r'$I_{\mathrm{e}}^{\mathrm{relax}}$',
        'I_e^work': r'$I_{\mathrm{e}}^{\mathrm{work}}$',
        'F': r'$\beta\left< F^{\mathrm{add}}(t) \right>$',
        'Delta F': r'$\beta \left< \Delta F_{\mathrm{neq}}^{\mathrm{relax}}(t)'
        ' \right>$',
        '-Delta F': r'$-\beta \left< \Delta F_{\mathrm{neq}}^{\mathrm{relax}}(t)'
        ' \right>$',
        'r(t)': r'$r (t)$',
        'eta(t)': r'$\eta (t)$',
        'mu(t)': r'$\mu (t)$',
        'W_diss(t)': r'$\beta \left< W_{\mathrm{diss}}(t) \right>$'
    }

    var_units = {
        ('H[Y_t]', 2): 'bits', ('H[Y_t]', euler): 'nats',
        ('H[X_t]', 2): 'bits', ('H[X_t]', euler): 'nats',
        ('H[Y_t|X_t]', 2): 'bits', ('H[Y_t|X_t]', euler): 'nats',
        ('H[X_t|Y_t]', 2): 'bits', ('H[X_t|Y_t]', euler): 'nats',
        ('H[Y_t|X_t+1]', 2): 'bits', ('H[Y_t|X_t+1]', euler): 'nats',
        ('H[X_t+1|Y_t]', 2): 'bits', ('H[X_t+1|Y_t]', euler): 'nats',
        ('H[X_t,Y_t]', 2): 'bits', ('H[X_t,Y_t]', euler): 'nats',
        ('H[X_t+1,Y_t]', 2): 'bits', ('H[X_t+1,Y_t]', euler): 'nats',
        ('I[X_t,Y_t]', 2): 'bits', ('I[X_t,Y_t]', euler): 'nats',
        ('I[X_t+1,Y_t]', 2): 'bits', ('I[X_t+1,Y_t]', euler): 'nats',
        ('sigma_X(t)', 2): 'bits', ('sigma_X(t)', euler): 'nats',
        ('sigma_Y(t)', 2): 'bits', ('sigma_Y(t)', euler): 'nats',
        ('sigma(t)', 2): 'bits', ('sigma(t)', euler): 'nats',
        ('l_Y(t)', 2): 'bits', ('l_Y(t)', 2): 'nats',
        ('I_nos(t)', 2): 'bits', ('I_nos(t)', euler): 'nats',
        ('T_{X->Y}(t)', 2): 'bits', ('T_{X->Y}(t)', euler): 'nats',
        ('Delta I', 2): 'bits', ('Delta I', euler): 'nats',
        ('Delta H[X]', 2): 'bits', ('Delta H[X]', euler): 'nats',
        ('Delta H[Y]', 2): 'bits', ('Delta H[Y]', euler): 'nats',
        ('I_e', 2): 'bits', ('I_e', euler): 'nats',
        ('I_e^relax', 2): 'bits', ('I_e^relax', euler): 'nats',
        ('I_e^work', 2): 'bits', ('I_e^work', euler): 'nats',
        ('F', 2): 'bits', ('F', euler): 'nats',
        ('Delta F', 2): 'bits', ('Delta F', euler): 'nats',
        ('-Delta F', 2): 'bits', ('-Delta F', euler): 'nats',
        ('r(t)', 2): '', ('r(t)', euler): '',
        ('eta(t)', 2): '', ('eta(t)', euler): '',
        ('mu(t)', 2): '', ('mu(t)', euler): '',
        ('W_diss(t)', 2): '', ('W_diss(t)', euler): ''
    }

    var_list = list(var_names.keys())

    return var_names, var_units, var_list

def read_from_config():
    """\
    Description:
    
    Inputs:

    Outputs:
    """

    config = ConfigParser()
    config.read(getcwd() + '/input.pyinp')

    return config['User Set Parameters']

def get_params(indices, n_grid, log=False, base=2):
    """\
    Description:

    Inputs:

    Outputs:
    """

    if log == True:
        log = tuple(True for index in indices)
    elif log == False:
        log = tuple(False for index in indices)
    else:
        assert len(log) == len(indices), \
            "Mismatch in the length of log and index!"
    
    pars = tuple(
        base**(ind_i-(n_grid-1)) if log[i] else (ind_i*(1.0/(n_grid-1)))
        for i, ind_i in enumerate(indices)
    )

    return pars

def safe_log(arg, log_base):
    """\
    Description:

    Inputs:

    Outputs:
    """
    with errstate(divide='ignore', invalid='ignore'):
        return log(arg).filled(0.0)/nlog(log_base)

def get_unit_label(unit_string):
    """\
    Description:

    Inputs:

    Outputs:
    """

    if unit_string == '':
        return ''
    else:
        return ' (' + unit_string + ')'




