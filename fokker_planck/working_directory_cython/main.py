from math import pi
from numpy import array, arange, empty, finfo, pi as npi
from matplotlib import cm, colors, rcParams, rc
from matplotlib.style import use as use
from matplotlib.pyplot import subplots, close, show

from fpe import launchpad
from fpe2 import launchpad_coupled


use('seaborn-paper')
rc('text', usetex=True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = [
    r"\usepackage{amsmath}", r"\usepackage{lmodern}", r"\usepackage{siunitx}", r"\usepackage{units}",
    r"\usepackage{physics}"
]

def get_params():
    gamma = 1000.0  # drag
    beta = 1.0  # 1/kT
    m = 1.0  # mass
    dt = 0.01  # time discretization

    ## system-specific parameters
    cycles = 1.0  # Max number of cycles
    N = 20
    write_out = 1./dt  # to have write out every second use 1/dt
    write_out = 1.0 # to write out every calculation

    # steady_state = True
    steady_state = False

    return (gamma, beta, m, dt, cycles, N, write_out, steady_state)

def save_data(times, amplitudes, trap_strengths, work_array, flux_array):

    rot_rate = 1./times

    target_dir = './output_dir/'
    for index1, A in enumerate(amplitudes):
        for index2, k in enumerate(trap_strengths):
            filename = '/output_file_k{0}_A{1}_final.dat'.format(k, A)
            with open(target_dir + filename, 'w') as ofile:
                for index3, t_and_f in enumerate(zip(times, rot_rate)):
                    t = t_and_f[0]
                    f = t_and_f[1]
                    ofile.write(
                        "{0}\t{1}\t{2}\t{3}\n".format(
                            t, f,
                            work_array[index1, index2, index3],
                            flux_array[index1, index2, index3]
                            )
                        )
                ofile.flush()

def main():

    # periods_to_run = 2**(arange(-8.0, 8.0))
    # amplitudes = array([0.0, 2.0, 4.0, 8.0])
    # trap_strengths = array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    periods_to_run = array([100.0])
    amplitudes = array([0.0])
    trap_strengths = array([40.0])

    works = empty((amplitudes.size, trap_strengths.size, periods_to_run.size))
    fluxes = empty((amplitudes.size, trap_strengths.size, periods_to_run.size))

    gamma, beta, m, dt, cycles, N, write_out, steady_state = get_params()

    dt = ((2*pi/N)*m*gamma / (1.5*amplitudes.max() + 0.5*trap_strengths.max()))
    dt_sim = dt / 10.0

    if steady_state:
        steady_state_var = 1
    else:
        steady_state_var = 0

    for index1, A in enumerate(amplitudes):
        for index2, k in enumerate(trap_strengths):
            for index3, period in enumerate(periods_to_run):

                print("A:", A, "k:", k, "T:", period)

                dt_sim = 0.1

                flux, mean_flux, work, heat, p_sum, p_now, p_equil = launchpad(
                    steady_state_var, cycles, N, write_out, period, A, k,
                    dt_sim, m, beta, gamma
                    )

                # plot(flux); show()
                print(work, heat, mean_flux)
                print(p_sum)

                works[index1, index2, index3] = work
                fluxes[index1, index2, index3] = mean_flux

                print("normalization check:", p_sum)
                assert(abs(p_sum-1.0) <= finfo('float32').eps), \
                    "WARNING: Normalization of distribution lost. Quitting."

    # save_data(periods_to_run, amplitudes, trap_strengths, works, fluxes)

def main2():

    # periods_to_run = 2**(arange(-8.0, 8.0))
    # amplitudes = array([0.0, 2.0, 4.0, 8.0])
    # trap_strengths = array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    periods_to_run = array([1.0])
    amplitudes_x = array([1.0])
    amplitudes_y = array([1.0])
    amplitudes_xy = array([1.0])

    # fluxes = empty((amplitudes.size, trap_strengths.size, periods_to_run.size))

    gamma, beta, m, dt, cycles, N, write_out, steady_state = get_params()

    # dt = ((2*pi/N)*m*gamma / (1.5*amplitudes.max() + 0.5*trap_strengths.max()))
    # dt_sim = dt / 10.0

    if steady_state:
        steady_state_var = 1
    else:
        steady_state_var = 0

    for index1, Axy in enumerate(amplitudes_xy):
        for index2, Ax in enumerate(amplitudes_x):
            for index3, Ay in enumerate(amplitudes_y):
                for period in periods_to_run:

                    print("Ax:", Ax, "Axy:", Axy, "Ay:", Ay, "T:", period)

                    dt_sim = 0.01

                    flux, mean_flux, work, heat, p_sum, p_now, p_equil, positions = launchpad_coupled(
                        steady_state_var, cycles, N, write_out, period, Ax, Axy, Ay,
                        dt_sim, m, beta, gamma
                    )

                    positions_deg = positions * (180.0/npi)

                    with open('joint_distribution.dat', 'w') as ofile:

                        for i in range(N):
                            for j in range(N):
                                ofile.write(str(p_now[i, j]) + '\t')
                            ofile.write('\n')
                        ofile.flush()

                    fig, ax = subplots(3, 1, figsize=(10,10), sharex='all', sharey='all')
                    cl0 = ax[0].contourf(positions_deg, positions_deg, p_now.T, 50)
                    ax[0].set_title(r'$\hat{\pi}^{\mathrm{eq}}$', fontsize=28)
                    ax[0].set_ylim([0, 360-(360/N)+0.001])
                    cl1 = ax[1].contourf(positions_deg, positions_deg, p_equil.T, 50)
                    ax[1].set_title(r'$\pi^{\mathrm{eq}}$', fontsize=28)
                    ax[1].set_ylim([0, 360-(360/N)+0.001])
                    cl2 = ax[2].contourf(positions_deg, positions_deg, (p_now / p_equil).T, 20)
                    ax[2].set_title(r'$\hat{\pi}^{\mathrm{eq}}/\pi^{\mathrm{eq}}$',fontsize=28)
                    ax[2].set_ylim([0, 360-(360/N)+0.001])
                    ax[0].tick_params(
                        axis='y', labelsize=22
                        )
                    ax[1].tick_params(
                        axis='y', labelsize=22
                        )
                    ax[2].tick_params(
                        axis='both', labelsize=22
                        )
                    fig.colorbar(cl0, ax=ax[0])
                    fig.colorbar(cl1, ax=ax[1])
                    fig.colorbar(cl2, ax=ax[2])

                    fig.tight_layout()
                    fig.text(
                        0.03, 0.51, r'$\theta_{y}$', va='center', ha='center',
                        fontsize=27, rotation='vertical'
                        )
                    fig.text(
                        0.47, 0.02, r'$\theta_{x}$', va='center', ha='center',
                        fontsize=27
                        )

                    left = 0.1  # the left side of the subplots of the figure
                    right = 1.0    # the right side of the subplots of the figure
                    bottom = 0.09   # the bottom of the subplots of the figure
                    top = 0.95      # the top of the subplots of the figure
                    wspace = 0.1  # the amount of width reserved for blank space between subplots
                    hspace = 0.20  # the amount of height reserved for white space between subplots

                    fig.subplots_adjust(
                        left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace
                        )

                    fig.savefig('probability_comparison.pdf')

                    print("Mean flux is = " + str(mean_flux))
                    fig2, ax2 = subplots(2, 1, figsize=(
                        10, 10), sharex='all', sharey='all')
                    cl0_1 = ax2[0].contourf(
                        positions_deg, positions_deg, flux[0].T, 50)
                    ax2[0].set_title(r'$\langle J_{1}(\vb{x})\rangle$', fontsize=28)
                    ax2[0].set_ylim([0, 360-(360/N)+0.001])
                    cl1_1 = ax2[1].contourf(
                        positions_deg, positions_deg, flux[1].T, 50)
                    ax2[1].set_title(r'$\langle J_{2}(\vb{x},t)\rangle$', fontsize=28)
                    ax2[1].set_ylim([0, 360-(360/N)+0.001])
                    ax2[0].tick_params(
                        axis='y', labelsize=22
                    )
                    ax2[1].tick_params(
                        axis='y', labelsize=22
                    )
                    ax2[1].tick_params(
                        axis='both', labelsize=22
                    )
                    fig2.colorbar(cl0_1, ax=ax2[0])
                    fig2.colorbar(cl1_1, ax=ax2[1])

                    fig2.tight_layout()
                    fig2.text(
                        0.03, 0.51, r'$\theta_{y}$', va='center', ha='center',
                        fontsize=27, rotation='vertical'
                    )
                    fig2.text(
                        0.47, 0.02, r'$\theta_{x}$', va='center', ha='center',
                        fontsize=27
                    )

                    left = 0.1  # the left side of the subplots of the figure
                    right = 1.0    # the right side of the subplots of the figure
                    bottom = 0.09   # the bottom of the subplots of the figure
                    top = 0.95      # the top of the subplots of the figure
                    wspace = 0.1  # the amount of width reserved for blank space between subplots
                    hspace = 0.20  # the amount of height reserved for white space between subplots

                    fig2.subplots_adjust(
                        left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace
                    )

                    fig2.savefig('flux_components.pdf')

                    close('all')

                    print("normalization check:", p_sum)
                    exit(0)

                    assert(abs(p_sum-1.0) <= finfo('float32').eps), \
                        "WARNING: Normalization of distribution lost. Quitting."

    # save_data(periods_to_run, amplitudes, trap_strengths, works, fluxes)

if __name__ == "__main__":
    # main()
    main2()
