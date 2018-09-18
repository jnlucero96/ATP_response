#!/bin/env python3
# Author: Joseph N. E. Lucero
# Date Created: 7 September 2018 12:11:52

# Import internal python libraries

# Import external python libraries
from numpy import einsum
# Import self-made libraries

def work_step(p_XY, work_matrix):
    """\
    Description:

    Inputs:

    Outputs:
    """
    return einsum('ij,jk', work_matrix, p_XY)

def relax_step(p_XY, relax_matrix):
    """\
    Description:

    Inputs:

    Outputs:
    """

    return einsum('ijk,ik->ij', relax_matrix, p_XY)



    

    


