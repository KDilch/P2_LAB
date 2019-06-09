#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.special import erfc
from scipy.optimize import curve_fit

def emg(xdata, h, mu, sigma, tau, c):
    return (h * sigma / tau) * np.sqrt(np.pi / 2) * np.exp(0.5 * (sigma / tau)**2 - (xdata - mu) / tau) * \
           erfc((1 / np.sqrt(2)) * (sigma / tau - (xdata - mu) / sigma)) + c
          
def fit_emg(xdata, ydata, initial_guess=None):
    return curve_fit(emg, xdata, ydata, initial_guess)

def get_mean_and_variance(h, mu, sigma, tau, c):
    return (mu + tau, sigma**2 + tau**2)