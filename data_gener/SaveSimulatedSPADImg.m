function SaveSimulatedSPADImg(fname, spad, SBR, range_bins, range_bins_hr, est_range_bins, rates_norm_params, rates, mean_signal_photons, mean_background_photons, bin_size)
%SaveSimulatedSPADImg Summary of this function goes here
%   * spad: sparse vector of the simulated photon counts
%   * SBR: Mean signal to background ration of the image
%   * bin: Depth image in units of time bins
%   * bin_hr: High-resolution Depth image in units of time bins
%   * est_range_bins: Estimated depth bins using some basic baselines (LMF,
%   ARGMAX, ZNCC)
%   * rates: Normalized Rates 3D ground truth photon flux rates from which spad where sampled.
%   * rates_norm_rates: Normalization params applied to rates. For some
%   reason storing the normalized rates occupies much less memory than the
%   un-normalized rates
%   * mean_signal_photons: Mean signal photons
%   * mean_background_photons: Mean background photons
%   * bin_size: Size of SPAD histogram time bin in seconds.

% save(fname, 'spad', 'SBR', 'range_bins', 'range_bins_hr', 'est_range_bins', 'rates', 'norm_rates', 'mean_signal_photons', 'mean_background_photons', 'bin_size');
% save(fname, 'spad', 'SBR', 'range_bins', 'range_bins_hr', 'rates', 'norm_rates', 'mean_signal_photons', 'mean_background_photons', 'bin_size');
% save(fname, 'spad', 'SBR', 'range_bins', 'range_bins_hr', 'norm_rates', 'mean_signal_photons', 'mean_background_photons', 'bin_size');

% For compatibility with previous code name the variables correctly
bin = range_bins; 
bin_hr = range_bins_hr;
photons = mean_signal_photons;
save(fname, 'spad', 'SBR', 'bin', 'range_bins', 'bin_hr', 'est_range_bins', 'rates_norm_params', 'rates', 'mean_signal_photons', 'mean_background_photons', 'bin_size', 'photons');

end
