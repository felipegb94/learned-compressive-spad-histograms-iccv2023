function [rates] = NormalizePhotonRates(rates)
% normalize the rate function to 0 to 1
rates = (rates - min(rates,[],3)) ./ (max(rates,[],3) - min(rates,[],3));
end