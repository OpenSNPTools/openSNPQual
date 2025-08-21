#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 18:32:05 2025

@author: dracid

This is a "namesake" function to load the sparams using scikit-rf
It's probably easier to just return rf.Network object, 
but for simplicity sake I keep the same format as IEEE370 code for now.

Input: touchstone *.snp file
Output: [freq, Sdata, npts] where
        freq - array of frequency points
        Sdata - sparam MATRIX form of numpy array
        npts - size of freq array
        
        
SPDX-License-Identifier: BSD-3-Clause
"""

import skrf as rf
import numpy as np
from typing import List, Dict, Tuple, Optional


def fromtouchn(filepath: str) -> Optional[rf.Network]:
    """Load touchstone file using scikit-rf"""
    try:
        network = rf.Network(filepath)
        
        freq = network.f
        npts = len(freq)
        Sdata = np.transpose(network.s, (1, 2, 0)) # Note: transposing the matrix matches the data structure used in IEEE370 code [m,n, freq]

        return freq, Sdata, npts
    
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None