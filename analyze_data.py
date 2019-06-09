import os
import numpy as np


DIR_NAME = "DATA"


def calculateAverageAndStdDev(vals, variances):
    mean = np.mean(vals)
    stdDev = np.sqrt(np.sum(variances)) / len(variances)
    return mean, stdDev


def calculateDeltaT(t2, stdDev2, t1, stdDev1):
    deltaT = t2 - t1
    stdDev = np.sqrt(stdDev1**2 + stdDev2**2)
    return deltaT, stdDev

def calculateVelocity(t, dist):
    """
    :param t:  in us
    :param dist: in cm
    :return: velocity in cm/us
    """
    vel = dist[0]/(t[0])
    uncertainty_vel = np.sqrt((dist[1]/t[0])**2 + (dist[0]*t[1]/t[0]**2)**2)
    return vel, uncertainty_vel

by_voltage = {}
for file in os.listdir(DIR_NAME):
    from_to, voltage = os.path.splitext(os.path.basename(file))[0].split("_")
    voltage = int(voltage)
    s1, v1, s2, v2 = np.loadtxt(os.path.join(DIR_NAME, file), unpack=True)
    mean1, stdDev1 = calculateAverageAndStdDev(s1, v1)
    mean2, stdDev2 = calculateAverageAndStdDev(s2, v2)
    deltaT, stdDev = calculateDeltaT(mean2, stdDev2, mean1, stdDev1)
    if not voltage in by_voltage:
        by_voltage[voltage] = {}
    by_voltage[voltage][from_to] = deltaT, stdDev
voltages = []
reduced_E_fields = []
t12s = []
t13s = []
t23s = []
v12s = []
v23s = []
v13s = []
pressure = 129700#[Pa]
dist12 = (4.60, 0.01) #[cm]
dist23 = (9.42, 0.01)
dist13 = (4.82, 0.01)

for voltage, times in by_voltage.items():
    voltages.append(voltage)
    reduced_E_fields.append(voltage/pressure)
    t13 = times["13"]
    t13s.append((t13))
    t23 = times["23"]
    t23s.append((t23))
    t12 = calculateDeltaT(t13[0], t13[1], t23[0], t23[1])
    t12s.append(t12)
    v12s.append(calculateVelocity(t12, dist12))
    v13s.append(calculateVelocity(t13, dist13))
    v23s.append(calculateVelocity(t23, dist23))


results = np.array([[voltage, reduced_E_field, t12[0], t12[1], t13[0], t13[1], t23[0], t23[1], v12[0], v12[1], v13[0], v13[1],v23[0], v23[1]] for voltage, reduced_E_field, t12, t13, t23, v12, v13, v23
                    in sorted(zip(voltages, reduced_E_fields, t12s, t13s, t23s, v12s, v13s, v23s))])
np.savetxt("results.tsv", results, delimiter="\t", header="Voltage (V)\t Reduced_E_field (V/Pa)\t t12 (us)\tstdDevt12 (us)\tt13 (us)\tstdDevt13 (us)\tt23 (us)\tstdDevt23 (us)\tVelocity12 (cm/s)\tStdbVelocity12 (cm/s)\tVelocity13 (cm/s)\tStdbVelocity13 (cm/s)\tVelocity23 (cm/s)\tStdbVelocity23 (cm/s)")
