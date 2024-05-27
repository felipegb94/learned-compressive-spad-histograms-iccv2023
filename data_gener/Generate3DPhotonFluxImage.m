function [rates] = Generate3DPhotonFluxImage(PSF_img, signal_ppp, ambient_ppp, range_bins, num_bins)
%Generate3DPhotonFluxImage Summary of this function goes here
%   Scale, vertically, and horizontally shift the PSF_img and add the
%   correct number of time bins.
    
    nr = size(PSF_img, 1);
    nc = size(PSF_img, 2);
    pulse_len = size(PSF_img, 3);

    % Verify input dims
    assert(size(signal_ppp,1)==nr, 'Incorrect dims in signal_ppp');
    assert(size(signal_ppp,2)==nc, 'Incorrect dims in signal_ppp');
    assert(size(ambient_ppp,1)==nr, 'Incorrect dims in ambient_ppp');
    assert(size(ambient_ppp,2)==nc, 'Incorrect dims in ambient_ppp');
    assert(size(range_bins,1)==nr, 'Incorrect dims in range_bins');
    assert(size(range_bins,2)==nc, 'Incorrect dims in range_bins');

    % construct the per-pixel ground truth photon flux signal
    
    % First, scale and add background photons to the PSF_img
    rates = zeros(nr, nc, num_bins);
    rates(:,:,1:pulse_len) = PSF_img;
    rates(:,:,1:pulse_len) = rates(:,:,1:pulse_len).*repmat(signal_ppp,[1,1,pulse_len]);
    rates = rates + repmat(ambient_ppp./num_bins,[1,1,num_bins]);  

    % find amount to circshift the rate function
    [~, pulse_max_idx] = max(PSF_img(1,1,:));
    circ_amount = range_bins - pulse_max_idx;
    for jj = 1:nr
        for kk = 1:nc
            rates(jj,kk,:) = circshift(squeeze(rates(jj,kk,:)), circ_amount(jj,kk));
        end
    end
    % TODO: add FOV with gaussian 2d filter
end
