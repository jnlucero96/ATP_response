#!/usr/bin/env python3

from math import pi
from numpy import (
    array, linspace, zeros, ones, diag, sin, cos, exp, arange, tan, kron,
    eye, union1d, where, meshgrid
)

# =================== TWO DIMENSIONAL PROBLEMS ==============================


class problem_2D(object):

    def __init__(
        self, x0=0.0, xn=2.0*pi,
        y0=0.0, ym=2.0*pi,
        n=360, m=360,
        E0=2.0, Ec=8.0, E1=2.0,
        num_minima0=3.0, num_minima1=3.0, phase=0.0
        D=0.001, psi0=0.0, psi1=0.0
    ):

        # unload the variables
        # define the end points of the intervals
        self.x0 = x0
        self.xn = xn
        self.y0 = y0
        self.ym = ym
        self.n = n  # number of points in x
        self.m = m  # number of points in y
        # barrier heights
        self.E0 = E0
        self.Ec = Ec
        self.E1 = E1
        # periodicity of potential
        self.num_minima0 = num_minima0
        self.num_minima1 = num_minima1
        # phasing of the different potentials
        self.phase = phase

        # diffusivity
        self.D = D

        # nonequilibrium forces
        self.psi0 = psi0
        self.psi1 = psi1

        # compute the derived variables

        # discretization
        self.L0 = xn-x0
        self.L1 = ym-y0
        self.dx = self.L0/self.n
        self.dy = self.L1/self.m

        # grid
        self.theta0 = linspace(x0, xn-self.dx, self.n)
        self.theta1 = linspace(y0, ym-self.dy, self.m)

        # drift matrices
        self.mu1 = self.drift1()
        self.mu2 = self.drift2()
        self.dmu1 = self.ddrift1()
        self.dmu2 = self.ddrift2()

        # potential landscape
        self.Epot = self.potential()

        # define equilibrium distribution
        self.p_equil = exp(-self.Epot)/exp(-self.Epot).sum()

    # define the potential V
    def potential(self):
        return 0.5*(
            self.E0*(1.0-cos(self.num_minima0*(self.theta0[:, None]-phase)))
            + self.Ec*(1.0-cos(self.theta0[:, None]-self.theta1[None, :]))
            + self.E1*(1.0-cos(self.num_minima1*self.theta1[None, :]))
        )

    # define drift vector mu_{1}
    def drift1(self):
        return -(
            0.5*(self.Ec*sin(self.theta0[:, None]-self.theta1[None, :])
                 + self.E0*self.num_minima0 *
                 sin(self.num_minima0*(self.theta0[:, None]-self.phase))
                 ) - self.psi0)

    # define drift vector mu_{2}
    def drift2(self):
        return -(
            0.5*(-self.Ec*sin(self.theta0[:, None]-self.theta1[None, :])
                 + self.E1*self.num_minima1 *
                 sin(self.num_minima1*self.theta1[None, :])
                 ) - self.psi1)

    # additional derivatives of the drift vectors above
    def ddrift1(self):
        return -(
            0.5*(self.Ec*cos(self.theta0[:, None]-self.theta1[None, :])
                 + self.E0*(self.num_minima0**2) *
                 cos(self.num_minima0*(self.theta0[:, None]-self.phase))
                 ))

    def ddrift2(self):
        return -(
            0.5*(self.Ec*cos(self.theta0[:, None]-self.theta1[None, :])
                 + self.E1*(self.num_minima1**2) *
                 cos(self.num_minima1*self.theta1[None, :])
                 ))
