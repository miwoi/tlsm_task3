

import tmlsm.data as td
import calibration_vs_test_in_time as fig_1
import stress_vs_strain as fig_2
import interpol_vs_extrapol as fig_3


E_infty = 0.5
E = 2.0
eta = 1.0
n = 100
omegas =[3.0]
As = [2.0]

eps, eps_dot, sig, dts = td.generate_data_harmonic(E_infty, E, eta, n, omegas, As)

print(sig.shape)

#fig_1.cal_test_time(sig,sig,sig,sig)
#fig_2.stress_strain_plot(sig,sig,sig,sig,eps,eps,eps,eps)
fig_3.interpol_extrapol(sig,sig,sig,sig)