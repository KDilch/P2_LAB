#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt

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
        means1 = []
        variances1 = []
        means2 = []
        variances2 = []
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
            mean1, var1 = get_mean_and_variance(*popt1)
            means1.append(mean1)
            variances1.append(var1)
            mean2, var2 = get_mean_and_variance(*popt2)
            means2.append(mean2)
            variances2.append(var2)
            # plt.plot(dataset[0], emg(dataset[0], *popt1), "b")
            # plt.plot(dataset[0], dataset[1], "b+")
            # plt.plot(dataset[0], emg(dataset[0], *popt2), "r")
            # plt.plot(dataset[0], dataset[2], "r+")
            # plt.savefig(os.path.join(dir_path + "_fitpng", f"{from_to}_{freq}_{index}.png"))
            # plt.clf()
            print(units, from_to, voltage, index)
        analyzed[from_to][str(voltage)] = np.array([means1, variances1, means2, variances2])

data_dir = "DATA"
shutil.rmtree(data_dir, ignore_errors=True)
os.mkdir(data_dir)
for from_to, byfreq in analyzed.items():
    for voltage, data in byfreq.items():
        np.savetxt(os.path.join(data_dir, f"{from_to}_{voltage}.tsv"), data.T, delimiter="\t", header="meanChannel1\tvarianceChannel1\tmeanChannel2\tvarianceChannel2")
print("done", analyzed)