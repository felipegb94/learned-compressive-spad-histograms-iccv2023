clear;

nr = 64;
nc = 64;
num_bins = 1024;

psf_img_param_idx = 1; % For bright image from which the PSF/pulse wavefor is extracted
[PSF_img1, psf1, pulse_len1] = LoadAndPreprocessBrightPSFImg(psf_img_param_idx, nr, nc, num_bins);

psf_img_param_idx = 0; 
[PSF_img2, psf2, pulse_len2] = LoadAndPreprocessBrightPSFImg(psf_img_param_idx, nr, nc, num_bins);

clf;
subplot(2,1,1);
r = randi(nr);
c = randi(nc);
plot(squeeze(PSF_img1(r,c,:)), 'linewidth', 3); hold on;
plot(squeeze(PSF_img1(10,10,:)), 'linewidth', 2);
plot(squeeze(PSF_img1(30,30,:)), 'linewidth', 1);
legend(sprintf('Row:%d,Col:%d',r,c), 'Row:10,Col:10', 'Row:30,Col:30')
xlim([0,pulse_len1])

subplot(2,1,2);
plot(squeeze(PSF_img2(r,c,:)), 'linewidth', 3); hold on;
plot(squeeze(PSF_img2(10,10,:)), 'linewidth', 2);
plot(squeeze(PSF_img2(30,30,:)), 'linewidth', 1);
legend(sprintf('Row:%d,Col:%d',r,c), 'Row:10,Col:10', 'Row:30,Col:30')
xlim([0,pulse_len1])
