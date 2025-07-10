#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 00:09:45 2025

@author: dracid
"""

import os
import numpy as np
from oct2py import Oct2Py
import matplotlib.pyplot as plt

# Run in CMD:
# export OCTAVE_EXECUTABLE=/var/lib/flatpak/exports/bin/org.octave.Octave
# os.environ["OCTAVE_EXECUTABLE"] = "/var/lib/flatpak/exports/bin/org.octave.Octave"


# https://www.perplexity.ai/search/on-ubuntu-24-04-i-have-install-IrE9mLPnRVqOq76Zq.fn3w
# Ubuntu, FLATPAK, run once: 
# os.environ["PATH"] = "/var/lib/flatpak/exports/bin/org.octave.Octave:" + os.environ["PATH"]
# then
# sudo flatpak override --filesystem=/tmp org.octave.Octave

oc = Oct2Py()

f = 2
x = np.linspace(0, 3, f*100)
y1 = np.sin(x*2*np.pi*f)
y2 = np.squeeze(oc.custom_sinx(x, f))

plt. plot(x,y1, x, y2)
