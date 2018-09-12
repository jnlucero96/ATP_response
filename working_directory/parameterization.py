#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 7 September 2018 09:45:30

# Import internal python libraries
from sys import exit

# Import external python libraries
# Import None

# Import self-made libraries
# Import None


def gamma1_gamma2_qbar_qtilde(pars, q_x, gamma_x):
    """\
    Description:

    Inputs:

    Outputs:
    """

    gamma_1 = pars[0]
    gamma_2 = pars[1]
    q_bar = pars[2]
    q_tilde = pars[3]

    if q_bar > 0.5:
        q_1 = 1-2*(1-q_bar)*(1-q_tilde)
    else:
        q_1 = 2*q_bar*q_tilde
    q_2 = 2*q_bar-q_1

    a = gamma_1*q_1
    b = gamma_1-a
    c = gamma_2*q_2
    d = gamma_2-c

    e = gamma_x*q_x
    f = gamma_x-e

    return a, b, c, d, e, f

def kplus_kminus_wplus_wminus(pars, q_x, gamma_x):
    """\
    Description:

    Inputs:

    Outputs:
    """

    k_plus = pars[0]
    k_minus = pars[1]
    w_plus = pars[2]
    w_minus = pars[3]

    a = k_plus
    b = k_minus
    c = w_minus
    d = w_plus
    e = q_x*gamma_x
    f = (1-q_x)*gamma_x

    return a, b, c, d, e, f

def kplus_kminus(pars, q_x, gamma_x):
    """\
    Description:

    Inputs:

    Outputs:
    """

    k_plus = pars[0]
    k_minus = pars[1]

    a = k_plus
    b = k_minus
    c = k_minus
    d = k_plus
    e = q_x*gamma_x
    f = (1-q_x)*gamma_x

    return a, b, c, d, e, f

def kplus_kminus_alpha(pars, q_x, gamma_x):
    """\
    Description:

    Inputs:

    Outputs:
    """

    k_plus = pars[0]
    k_minus = pars[1]
    alpha = pars[2]

    a = k_plus*alpha
    b = k_minus*alpha
    c = k_minus
    d = k_plus
    e = q_x*gamma_x
    f = (1-q_x)*gamma_x

    return a, b, c, d, e, f

def kplus_kminus_delta(pars, q_x, gamma_x):
    """\
    Description:

    Inputs:

    Outputs:
    """

    k_plus = pars[0]
    k_minus = pars[1]
    delta = pars[2]

    r = k_minus/k_plus

    a = k_plus
    b = k_plus*r
    c = k_plus*r**delta
    d = k_plus
    e = q_x*gamma_x
    f = (1-q_x)*gamma_x

    return a, b, c, d, e, f

def get_parameterization_funcs(q_x, gamma_x):
    """\
    Description:

    Inputs:

    Outputs:
    """

    return {
        'kplus_kminus': 
            lambda pars: kplus_kminus(pars, q_x, gamma_x),
        'kplus_kminus_alpha': 
            lambda pars: kplus_kminus_alpha(pars, q_x, gamma_x),
        'kplus_kminus_delta': 
            lambda pars: kplus_kminus_delta(pars, q_x, gamma_x),
        'kplus_kminus_wplus_wminus': 
            lambda pars: kplus_kminus_wplus_wminus(pars, q_x, gamma_x),
        'gamma1_gamma2_qbar_qtilde': 
            lambda pars: gamma1_gamma2_qbar_qtilde(pars, q_x, gamma_x)
    }


def select_parameterization(parameterization, q_x, gamma_x):
    """\
    Description:
    
    Inputs:

    Outputs:
    """

    parameterization_ref = [
        'kplus_kminus',
        'kplus_kminus_alpha',
        'kplus_kminus_delta',
        'kplus_kminus_wplus_wminus',
        'gamma1_gamma2_qbar_qtilde'
    ]

    param_funcs = get_parameterization_funcs(q_x, gamma_x)

    if not parameterization in parameterization_ref:
        print("Parametrization: " + parameterization + " was not found!")
        print("Parameterization must be one of:")
        for param in parameterization_ref:
            print(' - ' + param)
        exit(1)
    elif parameterization == 'kplus_kminus':
        log_params = (True, True)
        axes = (r'$k_{+}$', r'$k_{-}$')
    elif parameterization == 'kplus_kminus_alpha':
        log_params = (True, True, True)
        axes = (r'$k_{+}$', r'$k_{-}$', r'$\alpha$')
    elif parameterization == 'kplus_kminus_delta':
        log_params = (True, True, False)
        axes = (r'$k_{+}$', r'$k_{-}$', r'$\delta$')
    elif parameterization == 'kplus_kminus_wplus_wminus':
        log_params = (True, True, True, True)
        axes = (r'$k_{+}$', r'$k_{-}$', r'$w_{+}$', r'$w_{-}$')
    elif parameterization == 'gamma1_gamma2_qbar_qtilde':
        log_params = (True, True, True, False)
        axes = (r'$\gamma_{1}$', r'$\gamma_{2}$', r'$\bar{q}$', r'$\tilde{q}$')
    else:
        print(
            "WTF? I don't understand how this got through the if statement..."
            + " SOMETHING IS TERRIBLY WRONG!!!"
        )
        exit(1)

    return param_funcs[parameterization], log_params, axes

    
