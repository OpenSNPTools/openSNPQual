% Test script to see if this can be run in Python

function [y] = custom_sinx(x, freq)

  k = 2;
  y = sin(x.*2*pi*freq).*exp(-k.*x);


