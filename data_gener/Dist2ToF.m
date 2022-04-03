function [tof] = Dist2ToF(dist, c)
%Dist2ToF Convert distance to time-of-flight. dist and c should have
%consistent units
    tof = dist * 2 / c;
end