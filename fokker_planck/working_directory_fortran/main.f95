program main
implicit none

! this is what you think it is
real, parameter :: pi = 3.14159265358979323846264338327950288419716939937510582
! float32 machine eps
real, parameter :: float32_eps = 1.1920928955078125e-07
! float64 machine eps
real, parameter :: float64_eps = 2.22044604925031308084726e-16

! ============================================================================
! ===========
! =========== SIMULATION PARAMETERS
! ===========
! ============================================================================

! discretization parameters
real, parameter :: dt = 0.001  ! time discretization. Keep this number low
integer, parameter :: check_step = int(1.0/dt)
integer, parameter :: n = 360  ! inverse space discretization. Keep this number high!
real, parameter :: dx = (2.0*pi)/n

! model-specific parameters
real, parameter :: gamma = 1000.0  ! drag
real, parameter :: beta = 1.0  ! 1/kT
real, parameter :: m1 = 1.0  ! mass of system 1
real, parameter :: m2 = 1.0  ! mass of system 2

real, parameter :: E0 = 1.0 ! energy scale of F0 sub-system
real, parameter :: Ecouple = 1.0 ! energy scale of coupling between sub-systems F0 and F1
real, parameter :: E1 = 1.0 ! energy scale of F1 sub-system
real, parameter :: F_Hplus = 5.0 !  energy INTO (positive) F0 sub-system by H+ chemical bath
real, parameter :: F_atp = -5.0 ! energy INTO (positive) F1 sub-system by ATP chemical bath

real, parameter :: num_minima = 50.0 ! number of minima in the potential
real, parameter :: phase_shift = 0.0 ! how much sub-systems are offset from one another

! declare other variables to be used
real Z 

! declare iterator variables
integer i, j

! ============================================================================
! ===========
! =========== ARRAY INITIALIZATIONS
! ===========
! ============================================================================

! declare arrays to be used for simulation

real prob(n,n), p_now(n,n), p_last(n,n), p_last_ref(n,n), positions(n)
real potential_at_pos(n,n), force1_at_pos(n,n), force2_at_pos(n,n)
! real p_now_t(n,n), prob_t(n,n), potential_at_pos_t(n,n)
! real force1_at_pos_t(n,n), force2_at_pos_t(n,n)

positions = linspace(0.0, (2*pi)-dx, n)

do j=1,n
    do i=1,n
        potential_at_pos(i,j) = potential(positions(i), positions(j))
        force1_at_pos(i,j) = force1(positions(i), positions(j))
        force2_at_pos(i,j) = force2(positions(i), positions(j))
    end do
end do

! define the Gibbs-Boltzmann Equilibrium distribution
Z = sum(exp(-beta*potential_at_pos))
prob = exp(-beta*potential_at_pos)/Z

p_now = 1.0/(n*n)

call steady_state_initialize(p_now, p_last, p_last_ref, force1_at_pos, force2_at_pos)

! to ensure that it is in c order when written to file
p_now = transpose(p_now) 
prob = transpose(prob)
potential_at_pos = transpose(potential_at_pos)
force1_at_pos = transpose(force1_at_pos)
force2_at_pos = transpose(force2_at_pos)

open(unit=1, file="results.dat") ! open the file to write to

do j=1,n; do i=1,n
    write (*,"(5E24.16)") p_now(i,j), prob(i,j), potential_at_pos(i,j), &
        force1_at_pos(i,j), force2_at_pos(i,j)
enddo; enddo

close(1) ! close the file

contains

pure function force1(position1, position2); intent(in) position1, position2
    real position1, position2, force1
    force1 = (0.5)*( &
        Ecouple*sin(position1-position2) &
        + (num_minima*E0*sin((num_minima*position1)-(phase_shift))) &
        ) - F_Hplus
end function force1

pure function force2(position1, position2); intent(in) position1, position2
    real position1, position2, force2
    force2 = (0.5)*( &
        (-1.0)*Ecouple*sin(position1-position2) &
        + (num_minima*E1*sin(num_minima*position2)) &
        ) - F_atp
end function force2

pure function potential(position1, position2); intent(in) position1, position2
    real position1, position2, potential
    potential = 0.5*( &
        E0*(1-cos((num_minima*position1-phase_shift))) &
        + Ecouple*(1-cos(position1-position2)) & 
        +E1*(1-cos((num_minima*position2))) &
        )
end function potential

! Neumaier summation algorithm. Taken from wikipedia entry for 
! "Kahan summation algorithm"
function summation(array); intent(in) array
    real array(n,n), summation, c, t
    integer iii, jjj ! declare iterators 

    summation = 0.0; c = 0.0

    do jjj=1,n
        do iii=1,n 
            t = summation + array(iii,jjj)
            if (abs(summation) .GE. abs(array(iii,jjj))) then
                c = c + (summation - t) + array(iii,jjj)
            else
                c = c + (array(iii,jjj) - t) + summation 
            end if 
            summation = t
        end do
    end do

end function summation

function linspace(start, stop, n); intent(in) start, stop, n
    real start, stop
    integer n
    real delta, linspace(n)
    integer i

    delta = (stop-start)/(n-1.0)

    do i=1,n
        linspace(i) = start + (i*delta)
    end do
end function linspace

subroutine steady_state_initialize( &
    p_now, p_last, p_last_ref, force1_at_pos, force2_at_pos &
    )
    intent(in) force1_at_pos, force2_at_pos
    intent(inout) p_now, p_last, p_last_ref 

    real p_now(n,n), p_last(n,n), p_last_ref(n,n)
    real force1_at_pos(n,n), force2_at_pos(n,n)
    logical continue_condition
    integer step_counter
    real tot_var_dist

    continue_condition = .TRUE.
    step_counter = 0

    do while (continue_condition)
        ! save previous distribution and zero out current ones
        p_last = p_now; p_now = 0.0 

        call update_probability_full( &
            p_now, p_last, force1_at_pos, force2_at_pos &
            )

        ! bail at the first sign of trouble
        if (abs(sum(p_now) - 1.0) .ge. float32_eps) stop "Normalization broken!"
        
        if (step_counter == check_step) then
            tot_var_dist = 0.5*sum(abs(p_last_ref-p_now))

            write(*,*) tot_var_dist

            if (tot_var_dist < float64_eps) then
                continue_condition = .FALSE.
            else
                ! reset variables
                tot_var_dist = 0.0; step_counter = 0
                p_last_ref = p_now ! make current distribution reference
            end if
        end if

        step_counter = step_counter + 1 
    end do 

end subroutine steady_state_initialize

subroutine update_probability_full(p_now, p_last, force1_at_pos, force2_at_pos)

    ! declare size of incoming arrays
    real p_now(n,n), p_last(n,n)
    real force1_at_pos(n,n), force2_at_pos(n,n)

    ! declare iterators 
    integer i, j

    !! Periodic boundary conditions:
    !! Explicity update FPE for the corners
    p_now(1,1) = ( &
        p_last(1,1) &
        + dt*(force1_at_pos(2,1)*p_last(2,1)-force1_at_pos(n,1)*p_last(n,1))/(gamma*m1*2.0*dx) &
        + dt*(p_last(2,1)-2.0*p_last(1,1)+p_last(n,1))/(beta*gamma*m1*(dx*dx)) &
        + dt*(force2_at_pos(1,2)*p_last(1,2)-force2_at_pos(1,n)*p_last(1,n))/(gamma*m2*2.0*dx) &
        + dt*(p_last(1,2)-2.0*p_last(1,1)+p_last(1,n))/(beta*gamma*m2*(dx*dx)) &
        ) ! checked
    p_now(1,n) = ( &
        p_last(1,n) &
        + dt*(force1_at_pos(2,n)*p_last(2,n)-force1_at_pos(n,n)*p_last(n,n))/(gamma*m1*2.0*dx) &
        + dt*(p_last(2,n)-2.0*p_last(1,n)+p_last(n,n))/(beta*gamma*m1*(dx*dx)) &
        + dt*(force2_at_pos(1,1)*p_last(1,1)-force2_at_pos(1,n-1)*p_last(1,n-1))/(gamma*m2*2.0*dx) &
        + dt*(p_last(1,1)-2.0*p_last(1,n)+p_last(1,n-1))/(beta*gamma*m2*(dx*dx)) &
        ) ! checked
    p_now(n,1) = ( &
        p_last(n,1) &
        + dt*(force1_at_pos(1,1)*p_last(1,1)-force1_at_pos(n-1,1)*p_last(n-1,1))/(gamma*m1*2.0*dx) &
        + dt*(p_last(1,1)-2.0*p_last(n,1)+p_last(n-1,1))/(beta*gamma*m1*(dx*dx)) &
        + dt*(force2_at_pos(n,2)*p_last(n,2)-force2_at_pos(n,n)*p_last(n,n))/(gamma*m2*2.0*dx) &
        + dt*(p_last(n,2)-2.0*p_last(n,1)+p_last(n,n))/(beta*gamma*m2*(dx*dx)) &
        ) ! checked
    p_now(n,n) = ( &
        p_last(n,n) &
        + dt*(force1_at_pos(1,n)*p_last(1,n)-force1_at_pos(n-1,n)*p_last(n-1,n))/(gamma*m1*2.0*dx) &
        + dt*(p_last(1,n)-2.0*p_last(n,n)+p_last(n-1,n))/(beta*gamma*m1*(dx*dx)) &
        + dt*(force2_at_pos(n,1)*p_last(n,1)-force2_at_pos(n,n-1)*p_last(n,n-1))/(gamma*m2*2.0*dx) &
        + dt*(p_last(n,1)-2.0*p_last(n,n)+p_last(n,n-1))/(beta*gamma*m2*(dx*dx)) &
        ) !checked

    ! iterate through all the coordinates,not on the corners,for both variables
    do i=2,n-1
        !! Periodic boundary conditions:
        !! Explicitly update FPE for edges not corners
        p_now(1,i) = ( &
            p_last(1,i) &
            + dt*(force1_at_pos(2,i)*p_last(2,i)-force1_at_pos(n,i)*p_last(n,i))/(gamma*m1*2.0*dx) &
            + dt*(p_last(2,i)-2*p_last(1,i)+p_last(n,i))/(beta*gamma*m1*(dx*dx)) &
            + dt*(force2_at_pos(1,i+1)*p_last(1,i+1)-force2_at_pos(1,i-1)*p_last(1,i-1))/(gamma*m2*2.0*dx) &
            + dt*(p_last(1,i+1)-2*p_last(1,i)+p_last(1,i-1))/(beta*gamma*m2*(dx*dx)) &
            ) ! checked
        p_now(i,1) = ( &
            p_last(i,1) &
            + dt*(force1_at_pos(i+1,1)*p_last(i+1,1)-force1_at_pos(i-1,1)*p_last(i-1,1))/(gamma*m1*2.0*dx) &
            + dt*(p_last(i+1,1)-2*p_last(i,1)+p_last(i-1,1))/(beta*gamma*m1*(dx*dx)) &
            + dt*(force2_at_pos(i,2)*p_last(i,2)-force2_at_pos(i,n)*p_last(i,n))/(gamma*m2*2.0*dx) &
            + dt*(p_last(i,2)-2*p_last(i,1)+p_last(i,n))/(beta*gamma*m2*(dx*dx)) &
            ) ! checked

        !! all points with well defined neighbours go like so:
        do j=2,n-1
            p_now(i,j) = ( &
                p_last(i,j) &
                + dt*(force1_at_pos(i+1,j)*p_last(i+1,j)-force1_at_pos(i-1,j)*p_last(i-1,j))/(gamma*m1*2.0*dx) &
                + dt*(p_last(i+1,j)-2.0*p_last(i,j)+p_last(i-1,j))/(beta*gamma*m1*(dx*dx)) &
                + dt*(force2_at_pos(i,j+1)*p_last(i,j+1)-force2_at_pos(i,j-1)*p_last(i,j-1))/(gamma*m2*2.0*dx) &
                + dt*(p_last(i,j+1)-2.0*p_last(i,j)+p_last(i,j-1))/(beta*gamma*m2*(dx*dx)) &
                ) ! checked
        end do

        !! Explicitly update FPE for rest of edges not corners
        p_now(n,i) = ( &
            p_last(n,i) &
            + dt*(force1_at_pos(1,i)*p_last(1,i)-force1_at_pos(n-1,i)*p_last(n-1,i))/(gamma*m1*2.0*dx) &
            + dt*(p_last(1,i)-2.0*p_last(n,i)+p_last(n-1,i))/(beta*gamma*m1*(dx*dx)) &
            + dt*(force2_at_pos(n,i+1)*p_last(n,i+1)-force2_at_pos(n,i-1)*p_last(n,i-1))/(gamma*m2*2.0*dx) &
            + dt*(p_last(n,i+1)-2.0*p_last(n,i)+p_last(n,i-1))/(beta*gamma*m2*(dx*dx)) &
            ) ! checked
        p_now(i,n) = ( &
            p_last(i,n) &
            + dt*(force1_at_pos(i+1,n)*p_last(i+1,n)-force1_at_pos(i-1,n)*p_last(i-1,n))/(gamma*m1*2.0*dx) &
            + dt*(p_last(i+1,n)-2.0*p_last(i,n)+p_last(i-1,n))/(beta*gamma*m1*(dx*dx)) &
            + dt*(force2_at_pos(i,1)*p_last(i,1)-force2_at_pos(i,n-1)*p_last(i,n-1))/(gamma*m2*2.0*dx) &
            + dt*(p_last(i,1)-2.0*p_last(i,n)+p_last(i,n-1))/(beta*gamma*m2*(dx*dx)) &
            ) ! checked
    end do

end subroutine update_probability_full

end program main