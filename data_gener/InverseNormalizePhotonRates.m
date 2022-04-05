function [rates] = InverseNormalizePhotonRates(norm_rates, offset, scaling)
% normalize the rate function to 0 to 1
rates = (norm_rates./scaling) + offset;
end