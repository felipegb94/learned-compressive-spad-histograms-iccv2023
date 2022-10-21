function [dist] = ToF2Dist(tof, c)
%ToF2Dist
    dist = c*tof / 2;
end