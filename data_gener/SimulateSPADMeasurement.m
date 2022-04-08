function [spad, detections, rates, range_bins] = SimulateSPADMeasurement(albedo, intensity, dist, PSF_img, bin_size, num_bins, nr, nc, mean_signal_photons, mean_background_photons, dark_img, c)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


	% convert to time-of-flight
	tof = Dist2ToF(dist, c);
	
    % convert to bin number
	range_bins = ToF2Bins(tof, bin_size, num_bins, true);
	
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