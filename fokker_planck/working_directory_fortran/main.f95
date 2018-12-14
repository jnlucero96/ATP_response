program main
    implicit none

    real, parameter :: pi = 3.14159265358979323846264338327950288419716939937510582

    real, dimension (N, N) p_now

    print *, "F1=", force1(0.00, 2.00, 1.00, 1000.0, 1.0, 1.0)
    print *, "F2=", force2(0.00, 2.00, 1.00, 1000.0, 1.0, 1.0)

    do
        p_now[0, 0]=(
                p_last[0, 0]
                + dt*(force1(dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[1, 0] - force1(-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-1, 0])/(2.0*dx)
                + dt*(p_last[1, 0]-2.0*p_last[0, 0]+p_last[N-1, 0])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(0.0, dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, 1] - force2(0.0, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, N-1])/(2.0*dx)
                + dt*(p_last[0, 1]-2.0*p_last[0, 0]+p_last[0, N-1])/(beta*gamma*m2*(dx*dx))
                ) # checked
        p_now[0, N-1]=(
            p_last[0, N-1]
            + dt*(force1(dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[1, N-1] - force1(-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-1, N-1])/(2.0*dx)
            + dt*(p_last[1, N-1]-2.0*p_last[0, N-1]+p_last[N-1, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(0.0, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, 0] - force2(0.0, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, N-2])/(2.0*dx)
            + dt*(p_last[0, 0]-2.0*p_last[0, N-1]+p_last[0, N-2])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[N-1, 0]=(
            p_last[N-1, 0]
            + dt*(force1(0.0, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[0, 0] - force1(-2.0*dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-2, 0])/(2.0*dx)
            + dt*(p_last[0, 0]-2.0*p_last[N-1, 0]+p_last[N-2, 0])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(-dx, dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, 1] - force2(-dx, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, N-1])/(2.0*dx)
            + dt*(p_last[N-1, 1]-2.0*p_last[N-1, 0]+p_last[N-1, N-1])/(beta*gamma*m2*(dx*dx))
            ) # checked
        p_now[N-1, N-1]=(
            p_last[N-1, N-1]
            + dt*(force1(0.0, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[0, N-1] - force1(-2.0*dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-2, N-1])/(2.0*dx)
            + dt*(p_last[0, N-1]-2.0*p_last[N-1, N-1]+p_last[N-2, N-1])/(beta*gamma*m1*(dx*dx))
            + dt*(force2(-dx, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, 0] - force2(-dx, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, N-2])/(2.0*dx)
            + dt*(p_last[N-1, 0]-2.0*p_last[N-1, N-1]+p_last[N-1, N-2])/(beta*gamma*m2*(dx*dx))
            ) #checked

        ! iterate through all the coordinates, not on the corners, for both variables
        do i = 2, N
            ! Periodic boundary conditions:
            ! Explicitly update FPE for edges not corners
            p_now[0, i]=(
                p_last[0, i]
                + dt*(force1(dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[1, i] - force1(-dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-1, i])/(2.0*dx)
                + dt*(p_last[1, i]-2*p_last[0, i]+p_last[N-1, i])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(0.0, i*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, i+1] - force2(0.0, i*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[0, i-1])/(2.0*dx)
                + dt*(p_last[0, i+1]-2*p_last[0, i]+p_last[0, i-1])/(beta*gamma*m2*(dx*dx))
                ) # checked
            p_now[i, 0]=(
                p_last[i, 0]
                + dt*(force1(i*dx+dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i+1, 0] - force1(i*dx-dx, 0.0, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i-1, 0])/(2.0*dx)
                + dt*(p_last[i+1, 0]-2*p_last[i, 0]+p_last[i-1, 0])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(i*dx, dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, 1] - force2(i*dx, -dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, N-1])/(2.0*dx)
                + dt*(p_last[i, 1]-2*p_last[i, 0]+p_last[i, N-1])/(beta*gamma*m2*(dx*dx))
                ) # checked

            ! all points with well defined neighbours go like so:
            do j = 1, N-1
                p_now[i, j]= (
                    p_last[i, j]
                    + dt*(force1(i*dx+dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i+1, j] - force1(i*dx-dx, j*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i-1, j])/(2.0*dx)
                    + dt*(p_last[i+1, j]-2.0*p_last[i, j]+p_last[i-1, j])/(beta*gamma*m1*(dx*dx))
                    + dt*(force2(i*dx, j*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, j+1] - force2(i*dx, j*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, j-1])/(2.0*dx)
                    + dt*(p_last[i, j+1]-2.0*p_last[i, j]+p_last[i, j-1])/(beta*gamma*m2*(dx*dx))
                    ) # checked

            ! Explicitly update FPE for rest of edges not corners
            p_now[N-1, i]=(
                p_last[N-1, i]
                + dt*(force1(0.0, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[0, i] - force1(-2.0*dx, i*dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[N-2, i])/(2.0*dx)
                + dt*(p_last[0, i]-2.0*p_last[N-1, i]+p_last[N-2, i])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(-dx, i*dx+dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, i+1] - force2(-dx, i*dx-dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[N-1, i-1])/(2.0*dx)
                + dt*(p_last[N-1, i+1]-2.0*p_last[N-1, i]+p_last[N-1, i-1])/(beta*gamma*m2*(dx*dx))
                ) # checked
            p_now[i, N-1]=(
                p_last[i, N-1]
                + dt*(force1(i*dx+dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i+1, N-1] - force1(i*dx-dx, -dx, period, M_tot, m1, gamma, Ax, Axy, Ay)*p_last[i-1, N-1])/(2.0*dx)
                + dt*(p_last[i+1, N-1]-2.0*p_last[i, N-1]+p_last[i-1, N-1])/(beta*gamma*m1*(dx*dx))
                + dt*(force2(i*dx, 0.0, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, 0] - force2(i*dx, -2.0*dx, period, M_tot, m2, gamma, Ax, Axy, Ay)*p_last[i, N-2])/(2.0*dx)
                + dt*(p_last[i, 0]-2.0*p_last[i, N-1]+p_last[i, N-2])/(beta*gamma*m2*(dx*dx))
                ) # checked
        end do

    contains

    function force1(position1, position2, m1, gamma, Ax, Axy) result(f1)
        real, intent(in) :: position1, position2, m1, gamma, Ax, Axy
        real :: f1
        f1 = 0.5*(-1)*((1.0*Axy*sin(position1-position2)) + (3*Ax*sin((3*position1)-(2*pi/3))))/(2*gamma*m1)
    end function force1

    function force2(position1, position2, m2, gamma, Axy, Ay) result(f2)
        real, intent(in) :: position1, position2, m2, gamma, Axy, Ay
        real :: f2
        f2 = 0.5*((1.0*Axy*sin(position1-position2)) - (3*Ay*sin(3*position2)))/(2*gamma*m2)
    end function force2

end program main