import os
import numpy as np


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

sigma1_all = []
sigma2_all = []
by_voltage = {}
sigmas_by_voltage = {}
for file in os.listdir(DIR_NAME):
    from_to, voltage = os.path.splitext(os.path.basename(file))[0].split("_")
    voltage = int(voltage)
    s1, v1, s2, v2, sigmas1, sigmas2 = np.loadtxt(os.path.join(DIR_NAME, file), unpack=True)
    deltaTs = []
    variances = []
    for index in range(s1.shape[0]):
        deltaT, stdDev = calculateDeltaT(s1[index], 0.01/np.sqrt(3), s2[index], 0.01/np.sqrt(3))
        deltaTs.append(deltaT)
        variances.append(stdDev)
    mean, stdDev = calculateAverageAndStdDev(deltaTs, variances)
    sigma1, err_sigma1 = calculateAverageAndStdDev(sigmas1, variances)
    sigma1_all.append(sigma1)
    sigma2, err_sigma2 = calculateAverageAndStdDev(sigmas1, variances)
    sigma2_all.append(sigma2)
    if not voltage in by_voltage:
        by_voltage[voltage] = {}
    if not voltage in sigmas_by_voltage:
        sigmas_by_voltage[voltage] = {}
    by_voltage[voltage][from_to] = mean, stdDev
    sigmas_by_voltage[voltage][from_to] = sigma1, sigma2
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
sigma_12 = []
sigma_23 = []
sigma_13 = []
pressure = 129700#[Pa]
dist23 = (4.60, 0.01/np.sqrt(3)) #[cm]
dist13 = (9.42, 0.01/np.sqrt(3))
dist12 = (4.82, 0.01/np.sqrt(3))

for voltage, times in by_voltage.items():
    voltages.append(voltage)
    reduced_E_fields.append(voltage/pressure)
    t13 = times["13"]
    t13s.append(t13)
    t23 = times["23"]
    t23s.append(t23)
    t12 = calculateDeltaT(t13[0], t13[1], t23[0], t23[1])
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
    sigma_12.append(calc_width_of_spread(calc_diffusion_coeff(v12, dist12), t12))
    sigma_23.append(calc_width_of_spread(calc_diffusion_coeff(v23, dist23), t23))
    sigma_13.append(calc_width_of_spread(calc_diffusion_coeff(v13, dist13), t13))

sigmas1_13 = []
sigmas2_13 = []
sigmas1_23 = []
sigmas2_23 = []
for voltage, sigmas in sigmas_by_voltage.items():
    sigmas_13 = sigmas["13"]
    sigmas_23 = sigmas["23"]
    sigmas1_13.append(sigmas_13[0])
    sigmas1_23.append(sigmas_23[0])
    sigmas2_13.append(sigmas_13[1])
    sigmas2_23.append(sigmas_23[1])


print("reduced E field")
print(reduced_E_fields)
import matplotlib.pyplot as plt
plt.scatter(reduced_E_fields, sigmas1_13, label="D1")
plt.scatter(reduced_E_fields, sigmas2_23, label="D3")
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("Sigma (µs)")
plt.grid(True)
plt.savefig("rozmycie_czasowe_13.png")
plt.clf()

plt.scatter(reduced_E_fields, sigmas1_23, label='D2')
plt.scatter(reduced_E_fields, sigmas2_23, label='D3')
plt.legend()
plt.xlabel("E/p (V/Pa)")
plt.ylabel("Sigma (µs)")
plt.grid(True)
plt.savefig("rozmycie_czasowe_23.png")
plt.clf()


results = np.array([[voltage, reduced_E_field, t12[0], t12[1], t13[0], t13[1], t23[0], t23[1], v12[0], v12[1], v13[0], v13[1], v23[0], v23[1], v_av[0], v_av[1], d12[0], d12[1], d23[0], d23[1], d13[0], d23[1], s12[0], s12[1], s23[0], s23[1], s13[0], s13[1]] for voltage, reduced_E_field, t12, t13, t23, v12, v13, v23, v_av, d12, d13, d23, s12, s23, s13
                    in sorted(zip(voltages, reduced_E_fields, t12s, t13s, t23s, v12s, v13s, v23s, v_average, diffusion_12, diffusion_23, diffusion_13, sigma_12, sigma_23, sigma_13))])
np.savetxt("results.tsv", results, delimiter="\t", header="Voltage (V)\t Reduced_E_field (V/Pa)\t t12 (us)\tstdDevt12 (us)\tt13 (us)\tstdDevt13 (us)\tt23 (us)\tstdDevt23 (us)\tVelocity12 (cm/s)\tStdbVelocity12 (cm/s)\tVelocity13 (cm/s)\tStdbVelocity13 (cm/s)\tVelocity23 (cm/s)\tStdbVelocity23 (cm/s)\tv_av\tv_av_std\tdiffusion12\tstd_diffusion12\tdiffusion23\tstd_diffusion23\tdiffusion13\t std_diffusion13\tsigma_12\tstd_sigma12\tsigma23\tstd_sigma23\tsigma13\t std_sigma13")
