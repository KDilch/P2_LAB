import os
import numpy as np
from tabulate import tabulate


DIR_NAME = "DATA"


def calculateAverageAndStdDev(vals, variances):
    mean = np.mean(vals)
    stdDev = np.sqrt(np.sum(variances)) / len(variances)
    return mean, stdDev


def calculateDeltaT(t2, stdDev2, t1, stdDev1):
    deltaT = np.abs(t2 - t1)
    stdDev = np.sqrt(stdDev1**2 + stdDev2**2)
    return deltaT, stdDev

def calc_diffusion_coeff(v, s):
    diff_coeff = (1./3)*v[0]*s[0]
    err = np.sqrt(((1/3)*s[0]*v[1])**2 + ((1/3)*s[1]*v[0])**2)
    return diff_coeff, err

def calc_width_of_spread(diffusion, time):
    sigma = np.sqrt(2*diffusion[0]*time[0])
    err = np.sqrt((time[0]*diffusion[1]/np.sqrt(diffusion[0]))**2 + (diffusion[0]*time[1]/np.sqrt(time[0]))**2)
    return sigma, err


def calculateVelocity(t, dist):
    """
    :param t:  in us
    :param dist: in cm
    :return: velocity in cm/us
    """
    vel = dist[0]/(t[0])
    uncertainty_vel = np.sqrt((dist[1]/t[0])**2 + (dist[0]*t[1]/t[0]**2)**2)
    return vel, uncertainty_vel

tau_D3_13 = []
tau_D3_23 = []
Delta_tau_D3_13 = []
Delta_tau_D3_23 = []
by_voltage = {}
taus_D3_by_voltage = {}
for file in os.listdir(DIR_NAME):
    if '_tex' not in file:
        from_to, voltage = os.path.splitext(os.path.basename(file))[0].split("_")
        voltage = int(voltage)
        hs1, Delta_hs1, mus1, Delta_mus1, sigmas1, Delta_sigmas1, taus1, Delta_taus1, cs1, Delta_cs1, hs2, Delta_hs2, mus2, Delta_mus2, sigmas2, Delta_sigmas2, taus2, Delta_taus2, cs2, Delta_cs2 = np.loadtxt(os.path.join(DIR_NAME, file), unpack=True)
        deltaTs = []
        variances = []
        for index in range(mus1.shape[0]):
            deltaT, stdDev = calculateDeltaT(mus1[index], 0.01/np.sqrt(3), mus2[index], 0.01/np.sqrt(3))
            deltaTs.append(deltaT)
            variances.append(stdDev)
        mean, stdDev = calculateAverageAndStdDev(deltaTs, variances)
        tau_D3, err_tau_D3 = calculateAverageAndStdDev(taus2, Delta_taus2)
        if from_to=='13':
            tau_D3_13.append(tau_D3)
        elif from_to=='23':
            tau_D3_23.append(tau_D3)
        if not voltage in taus_D3_by_voltage:
            taus_D3_by_voltage[voltage] = {}
        taus_D3_by_voltage[voltage][from_to] = tau_D3, err_tau_D3
        if not voltage in by_voltage:
            by_voltage[voltage] = {}
        by_voltage[voltage][from_to] = mean, stdDev

voltages = []
reduced_E_fields = []
t12s = []
t13s = []
t23s = []
v12s = []
v23s = []
v13s = []
v_average = []
diffusion_12 = []
diffusion_23 = []
diffusion_13 = []
pressure = 129700#[Pa]
dist23 = (4.60, 0.01/np.sqrt(3)) #[cm]
dist13 = (9.42, 0.01/np.sqrt(3))
dist12 = (4.82, 0.01/np.sqrt(3))
t12s_arr = []

for voltage, times in by_voltage.items():
    voltages.append(voltage)
    reduced_E_fields.append(voltage/pressure)
    t13 = times["13"]
    t13s.append(t13)
    t23 = times["23"]
    t23s.append(t23)
    t12 = calculateDeltaT(t13[0], t13[1], t23[0], t23[1])
    t12s_arr.append(t12[0])
    t12s.append(t12)
    v12 = calculateVelocity(t12, dist12)
    v13 = calculateVelocity(t13, dist13)
    v23 = calculateVelocity(t23, dist23)
    v12s.append(v12)
    v13s.append(v13)
    v23s.append(v23)
    v_average.append(calculateAverageAndStdDev([v12[0], v13[0], v23[0]], [v12[1], v13[1], v23[1]]))
    diffusion_12.append(calc_diffusion_coeff(v12, dist12))
    diffusion_23.append(calc_diffusion_coeff(v23, dist23))
    diffusion_13.append(calc_diffusion_coeff(v13, dist13))

voltages1 = []
reduced_E_fields1 = []
taus_13 = []
Delta_taus_13 = []
taus_23 = []
Delta_taus_23 = []
tau_sq_diffs = []
for voltage, spread_D3 in taus_D3_by_voltage.items():
    voltages1.append(voltage)
    reduced_E_fields1.append(voltage / pressure)

    tau_13 = spread_D3["13"][0]
    Delta_tau_13 = spread_D3["13"][1]
    taus_13.append(tau_13)
    Delta_taus_13.append(Delta_tau_13)

    tau_23 = spread_D3["23"][0]
    Delta_tau_23 = spread_D3["23"][1]
    taus_23.append(tau_23)
    Delta_taus_23.append(Delta_tau_23)

    tau_sq_diffs.append((tau_13**2 - tau_23**2))

print("reduced E field")
print(reduced_E_fields)
import matplotlib.pyplot as plt
plt.errorbar(reduced_E_fields, taus_13, Delta_taus_13, fmt='.', label="D1-D3")
plt.errorbar(reduced_E_fields, taus_23, Delta_taus_23, fmt='.', label="D2-D3")
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("Tau (µs)")
plt.grid(True)
plt.savefig("rozmycie_czasowe_D3.png")
plt.clf()

plt.scatter(tau_sq_diffs, t12s_arr)
plt.xlabel("t12 (µs)")
plt.ylabel("tau13^2-tau23^2 (µs^2)")
plt.grid(True)
plt.savefig("diffusion.png")
plt.clf()


results = np.array([[voltage, reduced_E_field, t12[0], t12[1], t13[0], t13[1], t23[0], t23[1], v12[0], v12[1], v13[0], v13[1], v23[0], v23[1], v_av[0], v_av[1], d12[0], d12[1], d23[0], d23[1], d13[0], d23[1]] for voltage, reduced_E_field, t12, t13, t23, v12, v13, v23, v_av, d12, d13, d23
                    in sorted(zip(voltages, reduced_E_fields, t12s, t13s, t23s, v12s, v13s, v23s, v_average, diffusion_12, diffusion_23, diffusion_13))])
time_results = np.array([[voltage, reduced_E_field, t12[0], t12[1], t13[0], t13[1], t23[0], t23[1]] for voltage, reduced_E_field, t12, t13, t23
                    in sorted(zip(voltages, reduced_E_fields, t12s, t13s, t23s))])
velocity_results = np.array([[voltage, reduced_E_field, v12[0], v12[1], v13[0], v13[1], v23[0], v23[1]] for voltage, reduced_E_field, v12, v13, v23
                    in sorted(zip(voltages, reduced_E_fields, v12s, v13s, v23s))])
time_spread_results = np.array([[voltage, reduced_E_field, tau_13, Delta_tau_13, tau_23, Delta_tau_23] for voltage, reduced_E_field, tau_13, Delta_tau_13, tau_23, Delta_tau_23
                    in sorted(zip(voltages, reduced_E_fields, taus_13, Delta_taus_13, taus_23, Delta_taus_23))])
np.savetxt("results.tsv", results, delimiter="\t", header="Voltage (V)\t Reduced_E_field (V/Pa)\t t12 (us)\tstdDevt12 (us)\tt13 (us)\tstdDevt13 (us)\tt23 (us)\tstdDevt23 (us)\tVelocity12 (cm/s)\tStdbVelocity12 (cm/s)\tVelocity13 (cm/s)\tStdbVelocity13 (cm/s)\tVelocity23 (cm/s)\tStdbVelocity23 (cm/s)\tv_av\tv_av_std\tdiffusion12\tstd_diffusion12\tdiffusion23\tstd_diffusion23\tdiffusion13\t std_diffusion13\tsigma_12\tstd_sigma12\tsigma23\tstd_sigma23\tsigma13\t std_sigma13")
f = open(os.path.join(f"_tex_time_results.tex"), 'w+')
f.write(tabulate(time_results,
                                         headers=["Voltage (V)", "Reduced_E_field (V/Pa)", "t12 (us)", "Delta t12 (us)", "t13 (us)", " Delta t13 (us)", "t23 (us)", "Delta t23 (us)"],
                                         tablefmt='latex'))
f.close()
f1 = open(os.path.join(f"_tex_velocity_results.tex"), 'w+')
f1.write(tabulate(velocity_results,
                                         headers=["Voltage (V)", "Reduced_E_field (V/Pa)", "v12 (cm/s)", "Delta v12 (cm/s)", "v13 (cm/s)", "Delta v13 (cm/s)", "v23 (cm/s)", "Delta23 (cm/s)"],
                                         tablefmt='latex'))
f1.close()
f2 = open(os.path.join(f"_tex_time_spread_results.tex"), 'w+')
f2.write(tabulate(time_spread_results,
                                         headers=["Voltage (V)", "Reduced_E_field (V/Pa)", "tau13 (us)", "Delta tau13 (us)", "tau23 (us)", "Delta tau23 (us)"],
                                         tablefmt='latex'))
f2.close()