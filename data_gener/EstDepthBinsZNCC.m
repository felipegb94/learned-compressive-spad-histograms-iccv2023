function [est_depthbin] = EstDepthBinsZNCC(hist_img, psf)
%EstDepthBinsMatchedFilter Estimate the depth bin using zero-norm normalized cross-correlation
%   Compute argmax across histogram dimensin.
%   Assume that the last dimension is the histogram dimension

hist_img_dims = size(hist_img);
psf_dims = size(psf);
hist_img_last_dim = hist_img_dims(end);
psf_last_dim = psf_dims(end);

hist_img_len = hist_img_last_dim;
pulse_len = psf_last_dim;

% Vectorize the histogram image
vec_hist_img = VectorizeArray(hist_img);
num_px = size(vec_hist_img,1);
vec_psf = VectorizeArray(psf);
num_psf_elems = size(vec_psf, 1);

% Reshape psf if needed and replicate it
assert(pulse_len <= hist_img_len, 'Last dimenson of psf should be <= the last dim of hist image')
if(num_psf_elems > 1)
    assert(num_px == num_psf_elems, 'dims should match')
    assert(isequal(hist_img_dims(1:end-1), psf_dims(1:end-1)), "first n-1 dims should match")
else
    vec_psf = repmat(vec_psf, num_px, 1);
end

% Pad zeros to last dim of psf if needed
if(psf_last_dim < hist_img_last_dim)
    vec_psf(:, hist_img_len) = 0;
end

% Apply zero norm
zn_hist_img = ZeroNorm(vec_hist_img);
zn_psf = ZeroNorm(vec_psf);

% Compute Cirular X Correlation
zncc = CircXCorr(zn_psf, zn_hist_img);

% Get maximum bin
[~, est_depthbin] = max(zncc, [], 2);

% Remove offsets from PSF
[~, offset] = max(vec_psf, [], 2);
est_depthbin = est_depthbin + offset;

% est_depthbin = hist_img_len - reshape(est_depthbin, hist_img_dims(1:end-1));

est_depthbin = reshape(est_depthbin, hist_img_dims(1:end-1));

end

