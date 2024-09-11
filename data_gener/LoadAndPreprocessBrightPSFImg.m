function [PSF_img, psf, pulse_len] = LoadAndPreprocessBrightPSFImg(psf_img_param_idx, nr, nc, nbins)
%LoadAndPreprocessBrightPSFImg Generate a 3D PSF histogram image, where
%each pixel contains a histogram with the laser pulse waveform (IRF) for that
%pixel. The waveform may change from pixel to pixel, or it may be the same
%for all pixels
%   Parameters: 
%       * bright_img_param_idx: How to generate the PSF. If param_idx == 0,
%       then generate using Gaussian pulse, if param_idx == 1, load the
%       bright_img.mat
%       * nr, nc, nbins: Number of rows, cols, and timebins

    % 3D image with the laser waveform at each pixel
    
    if(psf_img_param_idx == 0)
        % This is what was used to simulate the Middlebury Test set in
        % Lindell et al. and Peng et al.
        disp("Generating PSF Image from Gaussian Pulse as in Middlebury Test set")
	% hard coding here, comes about 3.003
        pulse_len = (1760e-12) / ((600e-9) / 1024);
        pulse = normpdf(1:8*pulse_len,(8*pulse_len-1)/2,pulse_len/2);
        pulse = pulse ./ sum(pulse(:),1);
        PSF_img = repmat(reshape(pulse, [1, 1, numel(pulse)]),[nr,nc,1]);
        psf = pulse;
    elseif(psf_img_param_idx == 1)
        % This is what was used to simulate the NYUv2 Train set in
        % Lindell et al. and Peng et al.
        disp("Generating PSF Image from the ./TrainData/bright_img.mat data")

        % Load the PSF image which tells us the laser pulse waveform or IRF on each
        % pixel. The bright_img.mat is used to extract the laser pulse waveform for each pixel
        % individually. Since the data was obtained from a line laser we only need
        % data for a single line.
        load './TrainData/bright_img.mat' bright_img;
%         print_vector_stats(bright_img, "Bright Img");
        max_res = size(bright_img, 2);
        assert(nr<max_res, 'nr should be smaller the second dim of bright_img.mat');
        % Isolate psf
        res = nr;
        pulse_len = 16;
        bright_img = bright_img(:, 1:nr);
        [~,idx] = max(bright_img,[],1);
        psf = zeros(pulse_len, res);
        for ii = 1:res
           tmp = circshift(bright_img(:,ii), 10 - idx(ii));
           psf(:,ii) = flipud(tmp(1:pulse_len));
        end
        psf = psf ./ sum(psf,1); % Normalize
        psf(isnan(psf)) = 0;
%         print_vector_stats(psf, "PSF")

        % Replicate for the input cols and set the first few 
        pulse = repmat(psf,[1,1,nc]);       
        pulse = permute(pulse,[2,3,1]);
        PSF_img = pulse;
    elseif(psf_img_param_idx == 2)
        % This is a narrow IRF that we use to see if the model generalized
        % to other IRF
        disp("Generating PSF Image from Narrow Gaussian Pulse of 100ps width")
        bin_size = 100e-9 / nbins; % the temporal res
        pulse_len = (100e-12) / bin_size;
        pulse = normpdf(1:8*pulse_len,(8*pulse_len-1)/2,pulse_len/2);
        pulse = pulse ./ sum(pulse(:),1);
        PSF_img = repmat(reshape(pulse, [1, 1, numel(pulse)]),[nr,nc,1]);
        psf = pulse;
    elseif(psf_img_param_idx == 3)
        % This is a wide IRF that we use to see if the model generalized
        % to other IRF
        disp("Generating PSF Image from Wide Gaussian Pulse of 1000ps width")
        bin_size = 100e-9 / nbins; % the temporal res
        pulse_len = (1000e-12) / bin_size;
        pulse = normpdf(1:8*pulse_len,(8*pulse_len-1)/2,pulse_len/2);
        pulse = pulse ./ sum(pulse(:),1);
        PSF_img = repmat(reshape(pulse, [1, 1, numel(pulse)]),[nr,nc,1]);
        psf = pulse;
    else
        error('Bad psf_img_param_idx value');
    end
end
