#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:53:49 2025

@author: dracid
% Copyright (c) 2025, OpenSNPTools authors (See AUTHORS.md)

This is a Python adaptation of Octave code from IEEE 370 code:
https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG3/testQualityCheck_2Port.m
% Original MATLAB code copyright (c) 2017, IEEE 370 Open Source Authors (See IEEE370_AUTHORS.md)

"""

import fromtouchn
import qualityCheck
import qualityCheckFrequencyDomain


#%% Read Data
#TODO add scikit-rf parsing of sparams
[freq,Sdata,npts] = fromtouchn('../example_touchstone/xx.s2p'); 

#%% Settings
port_num = 2;
data_rate = 25.125; #data rate in Gbps
rise_per = 0.4; # rise time - fraction of UI
sample_per_UI = 32;

pulse_shape = 1; #1 is Gaussian; 2 is Rectangular with Butterworth filter; 3 is Rectangular with Gaussian filter;
extrapolation_method = 2; #1 is constant extrapolation; 2 is zero padding;

#%% Frequency domain checking
#TODO implement qualityCheckFrequencyDomain.py
[causality_metric_freq, reciprocity_metric_freq, passivity_metric_freq] = qualityCheckFrequencyDomain(Sdata,npts,port_num);

print([causality_metric_freq, passivity_metric_freq, reciprocity_metric_freq])

#%%Time domain checking
#TODO implement qualityCheck.py and dependencies
[causality_metric, reciprocity_metric, passivity_metric] = qualityCheck(freq,Sdata,port_num,data_rate,sample_per_UI,rise_per,pulse_shape,extrapolation_method,1);

print([causality_metric/2, passivity_metric/2, reciprocity_metric/2])