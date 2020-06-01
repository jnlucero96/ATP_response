#!/usr/bin/env python3

from numpy import loadtxt, zeros, linspace, pi
from scipy.interpolate import RectBivariateSpline


def compare_distr():

    target_dir = "/Users/jlucero/data_dir/2020-01-08/"
    reference_dir = target_dir + "/fd_reference_data/"

    Elst = [2.0]

    data_err = zeros(len(Elst))

    ref_theta = linspace(0.0, 2.0*pi-(2.0*pi/360), 360)
    target_theta = linspace(0.0, 2.0*pi-(2.0*pi/360), 360)

    fine_mesh = linspace(0.0, 2.0*pi-(2.0*pi/1000), 1000)

    for i, Ec in enumerate(Elst):
        file_name = (
            f"/reference_E0_2.0_Ecouple_{Ec}_E1_2.0_"
            + f"psi1_4.0_psi2_-2.0_"
            + f"n1_3.0_n2_3.0_phase_0.0_"
            + "outfile.dat"
        )

        ref_data = loadtxt(
            reference_dir + file_name, usecols=(0,)
        ).reshape((360, 360))
        target_data = loadtxt(
            target_dir + file_name, usecols=(0,)
        ).reshape((360, 360))

        # ref_data_interp = RectBivariateSpline(ref_theta, ref_theta, ref_data)
        ref_data_interp = RectBivariateSpline(
            target_theta, target_theta, ref_data)
        target_data_interp = RectBivariateSpline(
            target_theta, target_theta, target_data)

        # compute the inf-norm error between reference and new
        data_err[i] = (
            ref_data_interp(fine_mesh, fine_mesh) -
            target_data_interp(fine_mesh, fine_mesh)
        ).__abs__().max()

    print(ref_data.__abs__().max(), target_data.__abs__().max(),
          data_err)  # look at the data err


if __name__ == "__main__":
    compare_distr()
