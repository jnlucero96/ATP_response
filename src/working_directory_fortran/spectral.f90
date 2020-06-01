module fft_solve
implicit none

include "fftw3.f"

real(8), parameter :: float32_eps = 1.1920928955078125e-07
real(8), parameter :: float64_eps = 8.22044604925031308084726e-16
real(8), parameter :: pi = 4.0*atan(1.0)

contains

! propagate initial distribution until it reaches steady-state
subroutine get_spectral_steady( &
    n, m, dt, check_step, D, dx, dy, &
    mu1, dmu1, mu2, dmu2, p_initial, p_final &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: dt
    integer(8), intent(in) :: check_step
    real(8), intent(in) :: D, dx, dy
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2
    real(8), dimension(n,m), intent(in) :: p_initial
    real(8), dimension(n,m), intent(inout) :: p_final

    ! continue condition
    logical cc
    integer(8) step_counter ! counting steps
    real(8) tot_var_dist ! total variation distance

    complex(8), dimension(n,m) :: pr, phat_mat, p_penultimate, px_now
    complex(8), dimension(n*m+1) :: p_now
    real(8), dimension(n,m) :: p_last_ref
    integer(8) plan0, plan1, plan2

    real(8), dimension(:,:), allocatable :: kx, ky

    allocate ( kx(n,m), ky(n,m) )

    pr = p_initial

    ! planning is good
    call dfftw_plan_dft_2d(plan0,n,m,pr,phat_mat,FFTW_FORWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat_mat,px_now,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan2,n,m,phat_mat,p_penultimate,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! take the distribution into k-space
    call dfftw_execute_dft(plan0,pr,phat_mat)

    call dfftw_destroy_plan(plan0)

    ! initialize based on fourier transform of initial data
    p_now(:n*m) = pack(phat_mat, .TRUE.)
    p_now(n*m+1) = 0.0 ! set time = 0

    phat_mat = 0.0 ! reset the phat_mat matrix

    ! initialize reference array
    p_last_ref = 0.0

    ! initialize loop variables
    cc = .TRUE.; step_counter = 0

    ! get the frequencies ready
    kx = spread(fftfreq(n,dx),2,m)
    ky = spread(fftfreq(m,dy),1,n)

    do while (cc)

        ! update probabilities using specified scheme
        call imex(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p_now, D, dt)

        if (step_counter .EQ. check_step) then

            phat_mat = reshape(p_now(:n*m), [n,m])

            call dfftw_execute(plan1, phat_mat, px_now)

            ! check normalization and non-negativity of distributions
            ! bail at first sign of trouble
            if (abs(sum(realpart(px_now)/(n*m)) - 1.0) .ge. float32_eps) stop "Normalization broken!"
            if (count(realpart(px_now)/(n*m) < -float64_eps) .ge. 1) stop "Negative probabilities!"

            ! compute total variation distance
            tot_var_dist = 0.5*sum(abs(p_last_ref-(realpart(px_now)/(n*m))))

            if (tot_var_dist < float64_eps) then
                cc = .FALSE.
            else
                tot_var_dist = 0.0; step_counter = 0; phat_mat = 0.0
                p_last_ref = realpart(px_now)/(n*m)
            end if
        end if
        step_counter = step_counter + 1
    end do

    call dfftw_destroy_plan(plan1)

    phat_mat = reshape(p_now(:n*m), [n,m])

    p_penultimate = 0.0
    ! take the distribution back into real-space
    call dfftw_execute_dft(plan2,phat_mat,p_penultimate)
    call dfftw_destroy_plan(plan2)

    p_final = 0.0; p_final = realpart(p_penultimate)/(n*m)

    ! final checks on distribution before exiting subroutine
    ! bail if checks fail
    if (abs(sum(p_final) - 1.0) .ge. float32_eps) stop "Normalization broken!"
    if (count(p_final < -float64_eps) .ge. 1) stop "Negative probabilities!"
end subroutine get_spectral_steady

! wrapper for Crank-Nicolson Forward Euler (CNFT) scheme
subroutine imex(n, m, mu1, dmu1, mu2, dmu2, kx, ky, p, D, dt)
    integer(8), intent(in) :: n, m
    real(8), dimension(n,m), intent(in) :: mu1, dmu1, mu2, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(inout) :: p
    real(8), intent(in) :: D, dt
    complex(8), dimension(:), allocatable :: update

    allocate( update(n*m+1) )

    call evalRHS2(n, m, D, dt, mu1, mu2, dmu1, dmu2, kx, ky, p, update)

    p = update
end subroutine imex

! integrator for CNFT method
subroutine evalRHS2( &
    n, m, D, dt, &
    mu1, mu2, dmu1, dmu2, kx, ky, in_array, update &
    )
    integer(8), intent(in) :: n, m
    real(8), intent(in) :: D
    real(8), intent(in) :: dt
    real(8), dimension(n,m), intent(in) :: mu1, mu2, dmu1, dmu2, kx, ky
    complex(8), dimension(n*m+1), intent(in) :: in_array
    complex(8), dimension(n*m+1), intent(inout) :: update
    ! declare interior variables
    complex(8), dimension(n,m) :: pr, phat0, phat1, phat2, phat3
    complex(8), dimension(n,m) :: out0, out1, out2
    integer(8) plan0, plan1, plan2, plan3
    real(8) t
    ! process the input array
    pr = 0.0; pr = reshape(in_array(:n*m), [n,m])
    t = realpart(in_array(n*m+1))
    ! initialize real-space arrays
    out0 = 0.0; out1 = 0.0; out2 = 0.0
    ! initialize k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0; phat3 = 0.0
    ! plan all of the fft's that are to be done

    ! inverse fourier transforms back to real-space
    call dfftw_plan_dft_2d(plan0,n,m,pr,out0,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan1,n,m,phat1,out1,FFTW_BACKWARD,FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan2,n,m,phat2,out2,FFTW_BACKWARD,FFTW_ESTIMATE)

    ! forward fourier transforms to k-space
    call dfftw_plan_dft_2d(plan3,n,m,out0,phat0,FFTW_FORWARD,FFTW_ESTIMATE)

    phat1 = (0.0,1.0)*kx*pr
    if (modulo(n,2) .eq. 0) phat1(n/2+1,:) = 0.0

    phat2 = (0.0,1.0)*ky*pr
    if (modulo(m,2) .eq. 0) phat2(:,m/2+1) = 0.0

    phat3 = ((1.0/dt)-0.5*D*( kx**2+ky**2 ))*pr

    ! go back into real space
    call dfftw_execute_dft(plan0, pr, out0)
    call dfftw_execute_dft(plan1, phat1, out1)
    call dfftw_execute_dft(plan2, phat2, out2)

    ! do multiplications in real space
    out0 = -D*( &
        (dmu1+dmu2)*(realpart(out0)/(n*m)) &
        + mu1*(realpart(out1)/(n*m)) &
        + mu2*(realpart(out2)/(n*m)) &
        )

    ! reset the k-space arrays
    phat0 = 0.0; phat1 = 0.0; phat2 = 0.0

    ! go back into fourier space
    call dfftw_execute_dft(plan3, out0, phat0)

    call dfftw_destroy_plan(plan0)
    call dfftw_destroy_plan(plan1)
    call dfftw_destroy_plan(plan2)
    call dfftw_destroy_plan(plan3)

    update = 0.0
    update(:n*m) = pack((phat0 + phat3)/((1.0/dt)+0.5*D*( kx**2 + ky**2 )),.TRUE.)
    update(n*m+1) = 1.0
end subroutine evalRHS2

! get the frequencies in the correct order
! frequencies scaled appropriately to the length of the domain
function fftfreq(n, d)
    integer(8), intent(in) :: n
    real(8), intent(in) :: d
    real(8), dimension(n) :: fftfreq

    integer(8) i

    do i=1,n/2
        fftfreq(i) = (i-1)
        fftfreq(n+1-i) = -i
    end do

    fftfreq = (2.0*pi)*fftfreq/(d*n)
end function fftfreq
    
end module fft_solve