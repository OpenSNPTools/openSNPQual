clc
clear all
close all

f = 2
x = linspace(0, 3, f*100);
y = custom_sinx(x, f);

plot(x,y)
