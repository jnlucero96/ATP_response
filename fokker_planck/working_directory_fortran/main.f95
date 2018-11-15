program main
    implicit none

    real, parameter :: pi = 3.14159265358979323846264338327950288419716939937510582

    print *, "F1=", force1(0.00, 2.00, 1.00, 1000.0, 1.0, 1.0)
    print *, "F2=", force2(0.00, 2.00, 1.00, 1000.0, 1.0, 1.0)

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