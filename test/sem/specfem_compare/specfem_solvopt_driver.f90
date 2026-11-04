! SPECFEM2D's own nonlinear constant-Q fit (USE_SOLVOPT) for given quality factors: the
! relaxation times and strengths its attenuation_model computes, printed for
! run_compare_box.py (--specfem-solvopt) and the fit tests. Reads n_sls, f0 and the Q values
! from stdin (n_sls, f0, the count, then the values) and prints, per Q,
!     Q  tau_sigma_1 .. tau_sigma_N  w_1 .. w_N
! with tau_epsilon / tau_sigma - 1 = N w, over SPECFEM2D's band f0 / 10 .. 10 f0.
!
! Build against a SPECFEM2D build tree's objects (stop_the_code stubbed):
!     O=<specfem2d>/obj
!     gfortran -O2 -I$O specfem_solvopt_driver.f90 $O/attenuation_model.spec.o \
!         $O/shared_par.shared_module.o -o specfem_solvopt_driver
!     printf '3\n15\n2\n30\n10\n' | ./specfem_solvopt_driver
program specfem_solvopt_driver
  use shared_input_parameters, only: USE_SOLVOPT
  implicit none
  integer :: n, nq, iq, i
  double precision :: f0, q
  double precision, allocatable :: te(:), ts(:)
  read(*,*) n
  read(*,*) f0
  read(*,*) nq
  allocate(te(n), ts(n))
  USE_SOLVOPT = .true.
  do iq = 1, nq
    read(*,*) q
    call compute_attenuation_coeffs(n, q, f0, f0 / 10.d0, f0 * 10.d0, te, ts)
    write(*, '(ES25.17)', advance='no') q
    do i = 1, n
      write(*, '(ES25.17)', advance='no') ts(i)
    enddo
    do i = 1, n
      write(*, '(ES25.17)', advance='no') (te(i) / ts(i) - 1.d0) / n
    enddo
    write(*, *)
  enddo
end program specfem_solvopt_driver

subroutine stop_the_code(msg)
  character(len=*) :: msg
  print *, 'STOP: ', msg
  stop 1
end subroutine stop_the_code
