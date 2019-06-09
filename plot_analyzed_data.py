import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
(voltages, reduced_E_fields, t12s, stdDevT12s, t13s, stdDevT13s, t23s, stdDevT23s, v12s, stdDevV12s, v13s, stdDevV13s, v23s, stdDevV23s) = np.loadtxt("results.tsv", delimiter="\t", unpack=True)

palette = plt.get_cmap('Set1')

#================================Curve fits=============================================================================
def time_func(x, t0, a, b):
    return t0 + a*np.exp(-(1/b)*x)

def vel_func(x, a, b, c):
    return a + b*x + c*x**2

optimizedParameters_t12, pcov_t12 = opt.curve_fit(time_func, reduced_E_fields, t12s, sigma=stdDevT13s, p0=[2.5, 30, 0.005])
print(f"optimizedParameters_t12: t0:'{optimizedParameters_t12[0]}', a:'{optimizedParameters_t12[1]}', b:'{optimizedParameters_t12[2]}', \npcov: {pcov_t12}")
optimizedParameters_t13, pcov_t13 = opt.curve_fit(time_func, reduced_E_fields, t13s, sigma=stdDevT12s, p0=[4, 60, 0.005])
print(f"optimizedParameters_t13: t0:'{optimizedParameters_t13[0]}', a:'{optimizedParameters_t13[1]}', b:'{optimizedParameters_t13[2]}', \npcov: {pcov_t13}")
optimizedParameters_t23, pcov_t23 = opt.curve_fit(time_func, reduced_E_fields, t23s, sigma=stdDevT23s, p0=[2, 30, 0.005])
print(f"optimizedParameters_t23: t0:'{optimizedParameters_t23[0]}', a:'{optimizedParameters_t23[1]}', b:'{optimizedParameters_t23[2]}', \npcov: {pcov_t23}")


optimizedParameters_v12, pcov_v12 = opt.curve_fit(vel_func, reduced_E_fields, v12s, sigma=stdDevV12s, p0=[2.5, 30, 0.005])
print(f"optimizedParameters_v12: a:'{optimizedParameters_v12[0]}', b:'{optimizedParameters_v12[1]}', c:'{optimizedParameters_v12[2]}', \npcov: {pcov_v12}")
optimizedParameters_v13, pcov_v13 = opt.curve_fit(vel_func, reduced_E_fields, v13s, sigma=stdDevV13s, p0=[2.5, 30, 0.005])
print(f"optimizedParameters_v12: a:'{optimizedParameters_v13[0]}', b:'{optimizedParameters_v13[1]}', c:'{optimizedParameters_v13[2]}', \npcov: {pcov_v13}")
optimizedParameters_v23, pcov_v23 = opt.curve_fit(vel_func, reduced_E_fields, v23s, sigma=stdDevV23s, p0=[2.5, 30, 0.005])
print(f"optimizedParameters_v12: a:'{optimizedParameters_v23[0]}', b:'{optimizedParameters_v23[1]}', c:'{optimizedParameters_v23[2]}', \npcov: {pcov_v23}")
#================================Plots==================================================================================

plt.plot(reduced_E_fields, time_func(reduced_E_fields, *optimizedParameters_t12), color=palette(4), label="fit T 1,2")
plt.errorbar(reduced_E_fields, t12s, stdDevT12s, color=palette(4), fmt='.', label="T 1,2")
plt.plot(reduced_E_fields, time_func(reduced_E_fields, *optimizedParameters_t13), color=palette(1), label="fit T 1,3")
plt.errorbar(reduced_E_fields, t13s, stdDevT13s, color=palette(1), fmt='.', label="T 1,3")
plt.plot(reduced_E_fields, time_func(reduced_E_fields, *optimizedParameters_t23), color=palette(2), label="fit T 2,3")
plt.errorbar(reduced_E_fields, t23s, stdDevT23s, color=palette(2), fmt='.', label="T 2,3")
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("Time (µs)")
plt.savefig("results_plot_pos.png")
plt.clf()


plt.errorbar(reduced_E_fields, v12s, stdDevT12s, color=palette(4), fmt='.', label="V 1,2")
plt.plot(reduced_E_fields, vel_func(reduced_E_fields, *optimizedParameters_v12), color=palette(4), label="fit V 1,2")
plt.errorbar(reduced_E_fields, v13s, stdDevV13s, color=palette(1), fmt='.', label="V 1,3")
plt.plot(reduced_E_fields, vel_func(reduced_E_fields, *optimizedParameters_v13), color=palette(1), label="fit V 1,3")
plt.errorbar(reduced_E_fields, v23s, stdDevV23s, color=palette(2), fmt='.', label="V 2,3")
plt.plot(reduced_E_fields, vel_func(reduced_E_fields, *optimizedParameters_v23), color=palette(2), label="fit V 2,3")
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("Velocity (cm/µs)")
plt.savefig("results_plot_vel.png")
plt.clf()
