import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
(voltages, reduced_E_fields, t12s, stdDevT12s, t13s, stdDevT13s, t23s, stdDevT23s, v12s, stdDevV12s, v13s, stdDevV13s, v23s, stdDevV23s, v_av, std_v_av, diff_12, err_diff12, diff23, err_diff23, diff13, err_diff13) = np.loadtxt("results.tsv", delimiter="\t", unpack=True)

palette = plt.get_cmap('Set1')

#================================Curve fits=============================================================================
def time_func(x, t0, a):
    return t0 + a/x

def vel_func(x, a, b, c):
    return a + b*x + c*x**2

optimizedParameters_t12, pcov_t12 = opt.curve_fit(time_func, reduced_E_fields, t12s, sigma=stdDevT12s, p0=[3, 30])
print(f"optimizedParameters_t12: t0:'{optimizedParameters_t12[0]}', a:'{optimizedParameters_t12[1]}', \npcov: {pcov_t12}")
optimizedParameters_t13, pcov_t13 = opt.curve_fit(time_func, reduced_E_fields, t13s, sigma=stdDevT13s, p0=[4, 60])
print(f"optimizedParameters_t13: t0:'{optimizedParameters_t13[0]}', a:'{optimizedParameters_t13[1]}', \npcov: {pcov_t13}")
optimizedParameters_t23, pcov_t23 = opt.curve_fit(time_func, reduced_E_fields, t23s, sigma=stdDevT23s, p0=[2, 30])
print(f"optimizedParameters_t23: t0:'{optimizedParameters_t23[0]}', a:'{optimizedParameters_t23[1]}', \npcov: {pcov_t23}")
data_time = np.array([['t_12', optimizedParameters_t12[0], np.sqrt(pcov_t12[0][0]), optimizedParameters_t12[1], np.sqrt(pcov_t12[1][1])],
                     ['t_13', optimizedParameters_t13[0], np.sqrt(pcov_t13[0][0]), optimizedParameters_t13[1],
                      np.sqrt(pcov_t13[1][1])],
                     ['t_23', optimizedParameters_t23[0], np.sqrt(pcov_t23[0][0]), optimizedParameters_t23[1],
                      np.sqrt(pcov_t23[1][1])]])

optimizedParameters_v12, pcov_v12 = opt.curve_fit(vel_func, reduced_E_fields, v12s, sigma=stdDevV12s, p0=[2.5, 30, 0.005])
print(f"optimizedParameters_v12: a:'{optimizedParameters_v12[0]}', b:'{optimizedParameters_v12[1]}', c:'{optimizedParameters_v12[2]}', \npcov: {pcov_v12}")
optimizedParameters_v13, pcov_v13 = opt.curve_fit(vel_func, reduced_E_fields, v13s, sigma=stdDevV13s, p0=[2.5, 30, 0.005])
print(f"optimizedParameters_v13: a:'{optimizedParameters_v13[0]}', b:'{optimizedParameters_v13[1]}', c:'{optimizedParameters_v13[2]}', \npcov: {pcov_v13}")
optimizedParameters_v23, pcov_v23 = opt.curve_fit(vel_func, reduced_E_fields, v23s, sigma=stdDevV23s, p0=[2.5, 30, 0.005])
print(f"optimizedParameters_v23: a:'{optimizedParameters_v23[0]}', b:'{optimizedParameters_v23[1]}', c:'{optimizedParameters_v23[2]}', \npcov: {pcov_v23}")
data_velocity = np.array([['v_12', optimizedParameters_v12[0], np.sqrt(pcov_v12[0][0]), optimizedParameters_v12[1], np.sqrt(pcov_v12[1][1]), optimizedParameters_v12[2], np.sqrt(pcov_v12[2][2])],
                     ['v_13', optimizedParameters_v13[0], np.sqrt(pcov_v13[0][0]), optimizedParameters_v13[1],
                      np.sqrt(pcov_v13[1][1]), optimizedParameters_v13[2], np.sqrt(pcov_v13[2][2])],
                     ['v_23', optimizedParameters_v23[0], np.sqrt(pcov_v23[0][0]), optimizedParameters_v23[1],
                      np.sqrt(pcov_v23[1][1]), optimizedParameters_v23[2], np.sqrt(pcov_v23[2][2])]])
# ================================Plots=================================================================================

plt.plot(reduced_E_fields, time_func(reduced_E_fields, *optimizedParameters_t12), color=palette(4), label="fit T_12")
plt.errorbar(reduced_E_fields, t12s, stdDevT12s, color=palette(4), fmt='.', label="T_12")
plt.plot(reduced_E_fields, time_func(reduced_E_fields, *optimizedParameters_t13), color=palette(1), label="fit T_13")
plt.errorbar(reduced_E_fields, t13s, stdDevT13s, color=palette(1), fmt='.', label="T_13")
plt.plot(reduced_E_fields, time_func(reduced_E_fields, *optimizedParameters_t23), color=palette(2), label="fit T_23")
plt.errorbar(reduced_E_fields, t23s, stdDevT23s, color=palette(2), fmt='.', label="T_23")
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("Time (µs)")
plt.grid(True)
plt.savefig("results_plot_time_after_fixes.png")
plt.clf()


plt.errorbar(reduced_E_fields, v12s, stdDevT12s, color=palette(4), fmt='.', label="V_12")
plt.plot(reduced_E_fields, vel_func(reduced_E_fields, *optimizedParameters_v12), color=palette(4), label="fit V_12")
plt.errorbar(reduced_E_fields, v13s, stdDevV13s, color=palette(1), fmt='.', label="V_13")
plt.plot(reduced_E_fields, vel_func(reduced_E_fields, *optimizedParameters_v13), color=palette(1), label="fit V_13")
plt.errorbar(reduced_E_fields, v23s, stdDevV23s, color=palette(2), fmt='.', label="V_23")
plt.plot(reduced_E_fields, vel_func(reduced_E_fields, *optimizedParameters_v23), color=palette(2), label="fit V_23")
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("Velocity (cm/µs)")
plt.grid(True)
plt.savefig("results_plot_vel_after_fixes.png")
plt.clf()

plt.errorbar(reduced_E_fields, diff13, err_diff13, color=palette(4), fmt='.', label="D_13")
plt.errorbar(reduced_E_fields, diff_12, err_diff12, color=palette(1), fmt='.', label="D_12")
plt.errorbar(reduced_E_fields, diff23, err_diff23, color=palette(2), fmt='.', label="D_23")
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("D (mm^2/µs)")
plt.grid(True)
plt.savefig("diffusion.png")
plt.clf()

f = open(os.path.join(f"_tex_time_fit.tex"), 'w+')
f.write(tabulate(data_time,
                 headers=["Przypadek", "t0 (us)", "\Delta t_0 (us)", "a", "\Delta a"],
                                         tablefmt='latex'))
f.close()

f1 = open(os.path.join(f"_tex_vel_fit.tex"), 'w+')
f1.write(tabulate(data_velocity,
                 headers=["Przypadek", "a (us)", "Delta a (us)", "b", "Delta b", "c", "Delta c"],
                                         tablefmt='latex'))
f1.close()