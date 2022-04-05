% Load train instance, and compute depths
clear;

scene_id = 'living_room_0043a';
% scene_id = 'living_room_0042b';
scene_id = 'dining_room_0001a';
dataset_dirpath = './TrainData/SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1/';
scene_dirpath = fullfile(dataset_dirpath, scene_id);
files = dir(fullfile(scene_dirpath, 'spad_*_p1.mat'));
% files = dir(fullfile(scene_dirpath, 'spad_*_p3.mat'));
n_files = numel(files);
spad_data_fname = files(randi(n_files)).name;
% spad_data_fname = 'spad_0001_p3.mat';
spad_data_fpath = fullfile(scene_dirpath, spad_data_fname);

% Load data and PSF used;
data = load(spad_data_fpath);
psf_data = load(fullfile(dataset_dirpath, 'PSF_used_for_simulation.mat'));

% Set variables
[nr, nc, num_bins] = size(data.rates);
flux_norm_rates = data.rates;
flux_rates = InverseNormalizePhotonRates(flux_norm_rates, data.rates_norm_params.rates_offset, data.rates_norm_params.rates_scaling);
% flux_rates = data.rates;
PSF_img = psf_data.PSF_img;
range_bins = data.range_bins;
hist_sum = sum(flux_rates, 3);
spad_meas = reshape(full(data.spad), size(flux_rates));


mean_signal_photons = data.mean_signal_photons;
mean_background_photons = data.mean_background_photons;
fprintf('PhiSig: %f, PhiBkg: %f\n', mean_signal_photons, mean_background_photons);

% Test ZNCC Depth estimation

zncc_est_depthbin_on_flux = EstDepthBinsZNCC(flux_rates, PSF_img);
zncc_est_depthbin_on_spad = EstDepthBinsZNCC(spad_meas, PSF_img);
zncc_errs = zncc_est_depthbin_on_spad-range_bins;
zncc_abs_errs = abs(zncc_errs);
lmf_est_depthbin_on_flux = EstDepthBinsLogMatchedFilter(flux_rates, PSF_img);
lmf_est_depthbin_on_spad = EstDepthBinsLogMatchedFilter(spad_meas, PSF_img);
lmf_errs = lmf_est_depthbin_on_spad-range_bins;
lmf_abs_errs = abs(lmf_errs);
argmax_est_depthbin_on_flux = EstDepthBinsArgmax(flux_rates);
argmax_est_depthbin_on_spad = EstDepthBinsArgmax(spad_meas);
argmax_errs = argmax_est_depthbin_on_spad-range_bins;
argmax_abs_errs = abs(argmax_errs);

zMin = min(range_bins(:));
zMax = max(range_bins(:));



disp('Argmax:')
fprintf("    MAE = %f\n", mean(argmax_abs_errs(:)));
fprintf("    STDev = %f\n", std(argmax_abs_errs(:)));
disp('ZNCC:')
fprintf("    MAE = %f\n", mean(zncc_abs_errs(:)));
fprintf("    STDev = %f\n", std(zncc_abs_errs(:)));
disp('LM Filter:')
fprintf("    MAE = %f\n", mean(lmf_abs_errs(:)));
fprintf("    STDev = %f\n", std(lmf_abs_errs(:)));

clf;
subplot(4,3,1);
imagesc(range_bins); colorbar; caxis([zMin, zMax]); title('GT Depth Bins');
subplot(4,3,2);
imagesc(squeeze(sum(flux_rates,3))); colorbar; title('Total Flux');
subplot(4,3,3);
imagesc(squeeze(sum(spad_meas,3))); colorbar; title('Total Meas. Photons');

subplot(4,4,5);
imagesc(argmax_est_depthbin_on_flux); colorbar; caxis([zMin, zMax]); title('GT Flux (Argmax)');
subplot(4,4,6);
imagesc(argmax_est_depthbin_on_flux-range_bins); colorbar; caxis([-40,40]); title('Errs GT Flux (Argmax)');
subplot(4,4,7);
imagesc(argmax_est_depthbin_on_spad); colorbar; caxis([zMin, zMax]); title('SPAD (Argmax)');
subplot(4,4,8);
imagesc(argmax_errs); colorbar; caxis([-40,40]); title(sprintf('Errs SPAD (ARGMAX) | MAE=%.1f', mean(argmax_abs_errs(:))));

subplot(4,4,9);
imagesc(zncc_est_depthbin_on_flux); colorbar; caxis([zMin, zMax]); title('GT Flux (ZNCC)');
subplot(4,4,10);
imagesc(zncc_est_depthbin_on_flux-range_bins); colorbar; caxis([-40,40]); title('Errs GT Flux (ZNCC)');
subplot(4,4,11);
imagesc(zncc_est_depthbin_on_spad); colorbar; caxis([zMin, zMax]); title('SPAD (ZNCC)');
subplot(4,4,12);
imagesc(zncc_errs); colorbar; caxis([-40,40]); title(sprintf('Errs SPAD (ZNCC) | MAE=%.1f', mean(zncc_abs_errs(:))));

subplot(4,4,13);
imagesc(lmf_est_depthbin_on_flux); colorbar; caxis([zMin, zMax]); title('GT Flux (LMF)');
subplot(4,4,14);
imagesc(lmf_est_depthbin_on_flux-range_bins); colorbar; caxis([-40,40]); title('Errs GT Flux (LMF)');
subplot(4,4,15);
imagesc(lmf_est_depthbin_on_spad); colorbar; caxis([zMin, zMax]); title('SPAD (LMF)');
subplot(4,4,16);
imagesc(lmf_errs); colorbar; caxis([-40,40]); title(sprintf('Errs SPAD (LMF) | MAE=%.1f', mean(lmf_abs_errs(:))));

sgtitle(sprintf('%s \n PhiSig: %f, PhiBkg: %f',spad_data_fpath, mean_signal_photons, mean_background_photons)) 




