function [norm_rates, offset, scaling] = NormalizePhotonRates(rates)
% normalize the rate function to 0 to 1
EPSILON = 1e-8;
offset = min(rates,[],3);
scaling = 1 ./ (max(rates,[],3) - min(rates,[],3) + EPSILON);
norm_rates = (rates - offset) .* scaling;
% norm_rates = (rates - min(rates,[],3)) ./ (max(rates,[],3) - min(rates,[],3));
end