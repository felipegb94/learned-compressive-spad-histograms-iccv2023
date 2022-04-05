function [spad, detections, rates, range_bins, range_bins_hr] = SimulateSPADMeasurement(albedo_hr, intensity_hr, dist_hr, PSF_img, bin_size, num_bins, nr, nc, mean_signal_photons, mean_background_photons, dark_img, c)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

	% resize albedo and intensity to appropriate size
	albedo = imresize(albedo_hr, [nr nc], 'bilinear');
	intensity = imresize(intensity_hr, [nr nc], 'bilinear');
	dist = imresize(dist_hr, [nr nc], 'bilinear');

	% convert to time-of-flight
	tof = Dist2ToF(dist, c);
	tof_hr = Dist2ToF(dist_hr, c);
	
    % convert to bin number
	range_bins = ToF2Bins(tof, bin_size, num_bins, true);
	range_bins_hr = ToF2Bins(tof_hr, bin_size, num_bins, false);
	
    % set a number of signal photons per pixel
	alpha = albedo .* 1./ dist.^2;

	% add albedo/range/lighting/dark count effects
	[signal_ppp, ambient_ppp] = GeneratePhotonLevelImgs(alpha, intensity, mean_signal_photons, mean_background_photons, dark_img);
    
	% Scale and shift the PSFs to create the photon flux image
	rates = Generate3DPhotonFluxImage(PSF_img, signal_ppp, ambient_ppp, range_bins, num_bins);
				 
	% sample the process
	detections = poissrnd(rates);
	spad = sparse(reshape(detections, nr*nc, []));
		
end