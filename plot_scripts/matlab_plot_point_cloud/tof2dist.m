function [dist] = tof2dist(tof)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
c = 3e8;
dist = c*tof / 2;
end