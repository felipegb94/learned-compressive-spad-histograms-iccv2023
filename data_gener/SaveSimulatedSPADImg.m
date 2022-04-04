function SaveSimulatedSPADImg(fname, spad, SBR, range_bins, range_bins_hr, rates, norm_rates, mean_signal_photons, mean_background_photons, bin_size)
%SaveSimulatedSPADImg Summary of this function goes here
%   * spad: sparse vector of the simulated photon counts
%   * SBR: Mean signal to background ration of the image
%   * bin: Depth image in units of time bins
%   * bin_hr: High-resolution Depth image in units of time bins
%   * rates: 3D ground truth photon flux rates from which spad where sampled.
%   * norm_rates: Normalized Rates
%   * mean_signal_photons: Mean signal photons
%   * mean_background_photons: Mean background photons
%   * bin_size: Size of SPAD histogram time bin in seconds.
save(fname, 'spad', 'SBR', 'range_bins', 'range_bins_hr', 'rates', 'norm_rates', 'mean_signal_photons', 'mean_background_photons', 'bin_size');
end
