#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import curve_fit


def exponential(xdata, t0, A, b):
    return t0 + A*np.exp((-1/b) * xdata)


def fit_exponential(xdata, ydata, initial_guess=None):
    return curve_fit(exponential, xdata, ydata, initial_guess)


def quadratic(xdata, A, B, C):
    return A + B*xdata + C*np.power(xdata, 2)


def fit_quadratic(xdata, ydata, initial_guess=None):
    return curve_fit(quadratic, xdata, ydata, initial_guess)

