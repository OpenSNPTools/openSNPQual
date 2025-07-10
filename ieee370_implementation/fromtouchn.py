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

"""

import skrf as rf
import numy as np
from typing import List, Dict, Tuple, Optional


def fromtouchn(filepath: str) -> Optional[rf.Network]:
    """Load touchstone file using scikit-rf"""
    try:
        network = rf.Network(filepath)
        #TODO implement extraction of freq, Sdata and npts
        return network
    
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None