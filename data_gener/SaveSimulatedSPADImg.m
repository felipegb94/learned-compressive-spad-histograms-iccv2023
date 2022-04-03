function SaveSimulatedSPADImg(fname, spad, SBR, bin, bin_hr,  photons, rates)
%SaveSimulatedSPADImg Summary of this function goes here
%   * spad: sparse vector of the simulated photon counts
%   * SBR: Mean signal to background ration of the image
%   * bin: Depth image in units of time bins
%   * bin_hr: High-resolution Depth image in units of time bins
%   * photons: Mean signal photons
%   * rates: 3D photon flux rates.
    save(fname, 'spad', 'SBR', 'bin', 'bin_hr', 'photons', 'rates');
end
% 
%         % save sparse spad detections to file
%         % the 'spad' is the simulated data, 'depth' is the GT 2D depth map,
%         % which is 72*88, 'depth_hr' is 576*704. 'intensity' is the
%         % gray image, 72*88. 'rates' is actually the GT 3D histogram.
%         if LOW_RES
%             out_fname = sprintf('%s/LR_%s_%s_%s.mat', outdir, scenes{ss}, num2str(mean_signal_photons), num2str(mean_background_photons));
%             save(out_fname, 'spad', 'depth', 'SBR', 'mean_signal_photons', 'mean_background_photons', 'bin_size','intensity','intensity_hr', 'depth_hr', 'range_bins');
%         else
%             out_fname = sprintf('%s/HR_%s_%s_%s.mat',outdir, scenes{ss}, num2str(mean_signal_photons), num2str(mean_background_photons));
%             save(out_fname, 'spad', 'depth', 'SBR', 'mean_signal_photons', 'mean_background_photons', 'bin_size','intensity', 'range_bins', 'H', 'W');
%         end