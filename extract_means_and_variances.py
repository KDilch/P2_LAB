#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from read_data import read_data_set
from fit_data import emg, fit_emg, get_mean_and_variance

DIR_NAME = "KD"
ACCEPTANCE_RE = re.compile("^\d{2}_\d{3}\d?$")
INITIAL_GUESS_S1 = [1.73491118e+03, 3.53554040e-02, 7.67608013e-02, 1.15105759e+00, 0]
INITIAL_GUESS_S2 = [2.50655373e+02, 1.03182670e+01, 6.82937190e-02, 2.84176352e-01, 0]

analyzed = {"13": {}, "23": {}}

for dir in [name for name in os.listdir(DIR_NAME) if os.path.isdir(os.path.join(DIR_NAME, name))]:
    if ACCEPTANCE_RE.match(dir):
        dir_path = os.path.join(DIR_NAME, dir)
        shutil.rmtree(dir_path + "_fitpng", ignore_errors=True)
        os.mkdir(dir_path + "_fitpng")
        from_to, voltage, header, units, indices, data = read_data_set(dir_path)
        hs1 = []
        hs2 = []
        mus1 = []
        mus2 = []
        sigmas1 = []
        sigmas2 = []
        taus1 = []
        taus2 = []
        cs1 = []
        cs2 = []
        mus_diff = []
        Delta_hs1 = []
        Delta_hs2 = []
        Delta_mus1 = []
        Delta_mus2 = []
        Delta_sigmas1 = []
        Delta_sigmas2 = []
        Delta_taus1 = []
        Delta_taus2 = []
        Delta_cs1 = []
        Delta_cs2 = []
        Delta_mus_diff = []

        if (units[1] == "V"):
            data[:, 1, :] *= 1000
        if (units[2] == "V"):
            data[:, 2, :] *= 1000
        for index, dataset in zip(indices, data):
            try:
                popt1, pcov1 = fit_emg(dataset[0], dataset[1], INITIAL_GUESS_S1)
            except:
                print("could not fit 1!")
                continue
            try:
                INITIAL_GUESS_S2[1] = dataset[0][np.argmax(dataset[2])]
                popt2, pcov2 = fit_emg(dataset[0], dataset[2], INITIAL_GUESS_S2)
            except:
                print("could not fit 2!")
                continue
            hs1.append(popt1[0])
            hs2.append(popt2[0])
            mu1 = popt1[1]
            mu2 = popt2[1]
            mus1.append(popt1[1])
            mus2.append(popt2[1])
            sigmas1.append(popt1[2])
            sigmas2.append(popt2[2])
            Delta_hs1.append(np.sqrt(pcov1[0][0]))
            Delta_hs2.append(np.sqrt(pcov2[0][0]))
            Delta_mus1.append(np.sqrt(pcov1[1][1]))
            Delta_mus2.append(np.sqrt(pcov2[1][1]))
            Delta_sigmas1.append(np.sqrt(pcov1[2][2]))
            Delta_sigmas2.append(np.sqrt(pcov2[2][2]))
            tau1 = popt1[3]
            Delta_mu1 = np.sqrt(pcov1[1][1])
            Delta_mu2 = np.sqrt(pcov2[1][1])
            tau2 = popt2[3]
            taus1.append(tau1)
            taus2.append(tau2)
            Delta_taus1.append(np.sqrt(pcov1[3][3]))
            Delta_taus2.append(np.sqrt(pcov1[3][3]))
            cs1.append(popt1[4])
            cs2.append(popt2[4])
            Delta_cs1.append(np.sqrt(pcov1[4][4]))
            Delta_cs2.append(np.sqrt(pcov2[4][4]))
            if mu1 and mu2:
                mus_diff.append(np.abs(mu2-mu1))
                Delta_mus_diff.append(np.sqrt(Delta_mu1**2+Delta_mu2**2))
            else:
                mus_diff.append(0)
                Delta_mus_diff.append(0)
            # palette = plt.get_cmap('Blues')
            # palette1 = plt.get_cmap('Oranges')
            # plt.plot(dataset[0], emg(dataset[0], *popt1), color=palette(1000))
            # plt.scatter(dataset[0][::10],dataset[1][::10], color=palette(900))
            # # plt.errorbar(dataset[0], dataset[1], 0.01/np.sqrt(3), color=palette(900), fmt='.')
            # plt.plot(dataset[0], emg(dataset[0], *popt2), color=palette1(1000))
            # plt.scatter(dataset[0][::10],dataset[2][::10], color=palette1(900))
            # # plt.errorbar(dataset[0], dataset[2], 0.01/np.sqrt(3), color=palette1(900), fmt='.')
            # plt.savefig(os.path.join(dir_path + "_fitpng", f"{from_to}_{voltage}_{index}.png"))
            # plt.xlabel('Time (µs)')
            # plt.ylabel('Voltage (mV)')
            # plt.grid(True)
            # plt.show()
            # plt.clf()
            print(units, from_to, voltage, index)
        analyzed[from_to][str(voltage)] = np.array([hs1, Delta_hs1, mus1, Delta_mus1, sigmas1, Delta_sigmas1, taus1, Delta_taus1, cs1, Delta_cs1, hs2, Delta_hs2, mus2, Delta_mus2, sigmas2, Delta_sigmas2, taus2, Delta_taus2, cs2, Delta_cs2])

data_dir = "DATA"
shutil.rmtree(data_dir, ignore_errors=True)
os.mkdir(data_dir)
for from_to, byfreq in analyzed.items():
    for voltage, data in byfreq.items():
        np.savetxt(os.path.join(data_dir, f"{from_to}_{voltage}.tsv"), data.T, delimiter="\t", header="h1 [mV]\t Delta h1 [mV] \t mu1 [us] \t Delta_mu1 [us]\tsigma1 [us]\t Delta_sigma1 [us] \t tau1 [us]\t Delta_tau1 [us] \t c1 [mV] \t Delta_c1 [mV]\th2 [mV]\t Delta h2 [mV] \t mu2 [us] \t Delta_mu2 [us]\tsigma2 [us]\t Delta_sigma2 [us] \t tau2 [us]\t Delta_tau2 [us] \t c2 [mV] \t Delta_c2 [mV]")
        f = open(os.path.join(data_dir, f"_tex_{from_to}_{voltage}.tex"), 'w+')
        f.write(tabulate(data.T,
                                         headers=["h1 [mV]", "Delta h1 [mV]", "mu1 [us]", "Delta_mu1 [us]", "sigma1 [us]", "Delta_sigma1 [us]", "tau1 [us]", "Delta_tau1 [us]", "c1 [mV]", "Delta_c1 [mV]", "h2 [mV]", "Delta h2 [mV]", "mu2 [us]", "Delta_mu2 [us]", "sigma2 [us]", "Delta_sigma2 [us]", "tau2 [us]", "Delta_tau2 [us]", "c2 [mV]", "Delta_c2 [mV]"],
                                         tablefmt='latex'))
        f.close()
print("done", analyzed)
