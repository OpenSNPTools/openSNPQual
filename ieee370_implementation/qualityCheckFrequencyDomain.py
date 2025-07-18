#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 18:18:44 2025

@author: dracid
% Copyright (c) 2025, OpenSNPTools authors (See AUTHORS.md)

This is a Python adaptation of Octave code from IEEE 370 code:
https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG3/qualityCheckFrequencyDomain.m
% Original MATLAB code Copyright (c) 2017, IEEE 370 Open Source Authors (See IEEE370_AUTHORS.md)


Current version is based on the conversion:
https://www.codeconvert.ai/matlab-to-python-converter?id=9e751195-9592-4d2d-9299-5c2584df9aba


SPDX-License-Identifier: BSD-3-Clause
"""

import numpy as np
from typing import Tuple


def quality_check_frequency_domain(sdata: np.ndarray, nf: int, port_num: int) -> Tuple[float, float, float]:
    """
    IEEE P370 quality metrics calculation in frequency domain.
    
    This is a Python implementation of the IEEE P370 qualityCheckFrequencyDomain.m
    MATLAB function, maintaining the exact same algorithm and constants.
    
    Parameters
    ----------
    sdata : np.ndarray
        S-parameter data array of shape (port_num, port_num, nf)
    nf : int
        Number of frequency points
    port_num : int
        Number of ports
    
    Returns
    -------
    causality_metric_freq : float
        Causality metric (0-100, higher is better)
    reciprocity_metric_freq : float
        Reciprocity metric (0-100, higher is better)
    passivity_metric_freq : float
        Passivity metric (0-100, higher is better)
    
    Notes
    -----
    Original MATLAB implementation from IEEE P370 Working Group.
    """
    
    # Constants from original IEEE P370 implementation
    A = 1.00001  # Passivity threshold
    B = 0.1      # Weight factor
    C = 1e-6     # Reciprocity threshold
    
    # Initialize arrays
    PM = np.zeros(nf)  # Passivity metric
    PW = np.zeros(nf)  # Passivity weight
    RM = np.zeros(nf)  # Reciprocity metric
    RW = np.zeros(nf)  # Reciprocity weight
    
    # Passivity and Reciprocity calculations
    for i in range(nf):
        # Passivity: Check norm of S-matrix
        PM[i] = np.linalg.norm(sdata[:, :, i], ord=2) # MAKE SURE to use ord=2, since this is the Octave's default
        # numpy default norm() is "Frobenius", reference : https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        # Octave default norm() order p=2, reference: https://octave.sourceforge.io/octave/function/norm.html
        
        if PM[i] > A:
            PW[i] = (PM[i] - A) / B
        else:
            PW[i] = 0
        
        # Reciprocity: Check S(k,m) = S(m,k)
        RM[i] = 0
        for k in range(port_num):
            for m in range(port_num):
                RM[i] += abs(sdata[k, m, i] - sdata[m, k, i])
        
        # Normalize by number of off-diagonal elements
        Ns = port_num * (port_num - 1)
        RM[i] = RM[i] / Ns
        
        if RM[i] > C:
            RW[i] = (RM[i] - C) / B
        else:
            RW[i] = 0
    
    # Calculate final passivity and reciprocity metrics (0-100 scale)
    passivity_metric_freq = max((nf - np.sum(PW)), 0) / nf * 100
    reciprocity_metric_freq = max((nf - np.sum(RW)), 0) / nf * 100
    
    # Causality calculation
    CQM = np.zeros((port_num, port_num))
    
    for i in range(port_num):
        for j in range(port_num):
            total_r = 0
            positive_r = 0
            R = np.zeros(nf - 2)
            
            for k in range(nf - 2):
                # Calculate consecutive differences
                Vn = sdata[i, j, k + 1] - sdata[i, j, k]
                Vn1 = sdata[i, j, k + 2] - sdata[i, j, k + 1]
                
                # Cross product to check rotation direction
                R[k] = np.real(Vn1) * np.imag(Vn) - np.imag(Vn1) * np.real(Vn)
                
                if R[k] > 0:
                    positive_r += R[k]
                
                total_r += abs(R[k])
            
            # Calculate causality quality metric for this port pair
            if total_r > 0:
                CQM[i, j] = max(positive_r / total_r, 0) * 100
            else:
                CQM[i, j] = 100  # Perfect causality if no variation
    
    # Final causality metric is the minimum across all port pairs
    causality_metric_freq = np.min(CQM)
    
    return causality_metric_freq, reciprocity_metric_freq, passivity_metric_freq


def quality_check_frequency_domain_detailed(sdata: np.ndarray, nf: int, port_num: int) -> dict:
    """
    Extended version that returns detailed metrics for analysis.
    
    Parameters
    ----------
    sdata : np.ndarray
        S-parameter data array of shape (port_num, port_num, nf)
    nf : int
        Number of frequency points
    port_num : int
        Number of ports
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'causality': Causality metric (0-100)
        - 'reciprocity': Reciprocity metric (0-100)
        - 'passivity': Passivity metric (0-100)
        - 'passivity_violations': Array of frequency indices where passivity is violated
        - 'reciprocity_violations': Array of frequency indices where reciprocity is violated
        - 'causality_matrix': Full causality quality matrix for all port pairs
        - 'worst_causality_ports': Tuple of (i, j) for worst causality port pair
    """
    
    # Constants from IEEE P370
    A = 1.00001
    B = 0.1
    C = 1e-6
    
    # Arrays for detailed tracking
    PM = np.zeros(nf)
    PW = np.zeros(nf)
    RM = np.zeros(nf)
    RW = np.zeros(nf)
    passivity_violations = []
    reciprocity_violations = []
    
    # Passivity and Reciprocity
    for i in range(nf):
        # Passivity
        PM[i] = np.linalg.norm(sdata[:, :, i])
        if PM[i] > A:
            PW[i] = (PM[i] - A) / B
            passivity_violations.append(i)
        else:
            PW[i] = 0
        
        # Reciprocity
        RM[i] = 0
        for k in range(port_num):
            for m in range(port_num):
                RM[i] += abs(sdata[k, m, i] - sdata[m, k, i])
        
        Ns = port_num * (port_num - 1)
        RM[i] = RM[i] / Ns
        
        if RM[i] > C:
            RW[i] = (RM[i] - C) / B
            reciprocity_violations.append(i)
        else:
            RW[i] = 0
    
    # Final metrics
    passivity_metric_freq = max((nf - np.sum(PW)), 0) / nf * 100
    reciprocity_metric_freq = max((nf - np.sum(RW)), 0) / nf * 100
    
    # Causality with detailed matrix
    CQM = np.zeros((port_num, port_num))
    
    for i in range(port_num):
        for j in range(port_num):
            total_r = 0
            positive_r = 0
            
            for k in range(nf - 2):
                Vn = sdata[i, j, k + 1] - sdata[i, j, k]
                Vn1 = sdata[i, j, k + 2] - sdata[i, j, k + 1]
                R = np.real(Vn1) * np.imag(Vn) - np.imag(Vn1) * np.real(Vn)
                
                if R > 0:
                    positive_r += R
                total_r += abs(R)
            
            if total_r > 0:
                CQM[i, j] = max(positive_r / total_r, 0) * 100
            else:
                CQM[i, j] = 100
    
    causality_metric_freq = np.min(CQM)
    worst_ports = np.unravel_index(np.argmin(CQM), CQM.shape)
    
    return {
        'causality': causality_metric_freq,
        'reciprocity': reciprocity_metric_freq,
        'passivity': passivity_metric_freq,
        'passivity_violations': np.array(passivity_violations),
        'reciprocity_violations': np.array(reciprocity_violations),
        'causality_matrix': CQM,
        'worst_causality_ports': worst_ports
    }


# Integration with OpenSNPQual
class IEEEP370QualityMetrics:
    """
    IEEE P370 compliant quality metrics calculator.
    
    This class provides the official IEEE P370 implementation of S-parameter
    quality metrics as defined in the IEEE P370 standard.
    """
    
    def __init__(self):
        """Initialize with IEEE P370 constants."""
        self.A = 1.00001  # Passivity threshold
        self.B = 0.1      # Weight factor  
        self.C = 1e-6     # Reciprocity threshold
    
    def evaluate_network(self, network):
        """
        Evaluate a scikit-rf Network object using IEEE P370 metrics.
        
        Parameters
        ----------
        network : skrf.Network
            S-parameter network object
        
        Returns
        -------
        dict
            Quality metrics including causality, reciprocity, and passivity
        """
        # Extract S-parameter data
        sdata = network.s
        nf = len(network.f)
        port_num = network.nports
        
        # Transpose to match MATLAB convention (ports, ports, freq)
        sdata_transposed = np.transpose(sdata, (1, 2, 0))
        
        # Calculate metrics using IEEE P370 algorithm
        causality, reciprocity, passivity = quality_check_frequency_domain(
            sdata_transposed, nf, port_num
        )
        
        # Get detailed metrics
        detailed = quality_check_frequency_domain_detailed(
            sdata_transposed, nf, port_num
        )
        
        return {
            'passivity_freq': (100 - passivity) / 100,  # Convert to violation metric
            'reciprocity_freq': (100 - reciprocity) / 100,
            'causality_freq': (100 - causality) / 100,
            'ieee_p370_detailed': detailed
        }


# Example usage
if __name__ == "__main__":
    # Create example S-parameter data
    nf = 201  # Number of frequency points
    port_num = 2  # 2-port network
    
    # Generate example data (replace with actual S-parameter loading)
    freq = np.linspace(0, 20e9, nf)
    sdata = np.zeros((port_num, port_num, nf), dtype=complex)
    
    # Example: transmission line with some loss
    for i in range(nf):
        # S11, S22 (reflection)
        sdata[0, 0, i] = 0.1 * np.exp(1j * 2 * np.pi * freq[i] / 1e9)
        sdata[1, 1, i] = 0.1 * np.exp(1j * 2 * np.pi * freq[i] / 1e9)
        
        # S21, S12 (transmission) - reciprocal
        sdata[0, 1, i] = 0.9 * np.exp(-1j * 2 * np.pi * freq[i] / 5e9)
        sdata[1, 0, i] = 0.9 * np.exp(-1j * 2 * np.pi * freq[i] / 5e9)
    
    # Calculate metrics
    causality, reciprocity, passivity = quality_check_frequency_domain(sdata, nf, port_num)
    
    print(f"IEEE P370 Quality Metrics:")
    print(f"Causality:   {causality:.2f}%")
    print(f"Reciprocity: {reciprocity:.2f}%")
    print(f"Passivity:   {passivity:.2f}%")
    
    # Get detailed metrics
    detailed = quality_check_frequency_domain_detailed(sdata, nf, port_num)
    print(f"\nDetailed Analysis:")
    print(f"Passivity violations at {len(detailed['passivity_violations'])} frequency points")
    print(f"Reciprocity violations at {len(detailed['reciprocity_violations'])} frequency points")
    print(f"Worst causality between ports {detailed['worst_causality_ports']}")