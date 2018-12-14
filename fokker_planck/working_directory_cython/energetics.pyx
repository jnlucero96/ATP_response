# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
from libc.math cimport sin, cos

# yes, this is what you think it is
cdef double pi = 3.14159265358979323846264338327950288419716939937510582

cdef double force1(
    double position1, double position2,
    double Ax, double Axy, double H
    ) nogil: # force on system X
    return (0.5)*(Axy*sin(position1-position2)+(3*Ax*sin((3*position1)-(2*pi/3)))) + H

cdef double force2(
    double position1, double position2,
    double Ay, double Axy, double A
    ) nogil: # force on system Y
    return (0.5)*((-1.0)*Axy*sin(position1-position2)+(3*Ay*sin(3*position2))) - A

cdef double potential(
    double position1, double position2,
    double Ax, double Axy, double Ay
    ) nogil:
    return 0.5*(Ax*(1-cos((3*position1)-(2*pi/3)))+Axy*(1-cos(position1-position2))+Ay*(1-cos((3*position2))))