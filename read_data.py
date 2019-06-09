#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np

def parse_file(filename):
    with open(filename, "r") as tsvfile:
        index = os.path.splitext(os.path.basename(filename))[0].split("_")[2]
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        headers = np.array(next(tsvreader))
        units = np.array(list(map(lambda x: x.strip('()'), next(tsvreader))))
        # ignore Blank line
        next(tsvreader)
        data = np.array(list(zip(*list(tsvreader)))).astype(float)
        return (index, headers, units, index, data)

def read_data_set(dirname):
    setMeta = False
    headers = None
    units = None
    first, second = os.path.basename(dirname).split("_")
    between = first
    voltage = int(second)
    indices =[]
    dataset = []
    for file in os.listdir(dirname):
        if file.endswith(".txt"):
            (index, h, u, index, data) = parse_file(os.path.join(dirname, file))
            if (not setMeta):
                headers = h
                units = u
                setMeta = True
            indices.append(index)
            dataset.append(data)
    return (between, voltage, headers, units, np.array(indices), np.array(dataset))
            
    
