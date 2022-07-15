clear; close all;

visualize_data = false;
% Add paths
addpath('./TestData/middlebury/nyu_utils');

% Set paths
base_dirpath = './TestData/middlebury';
dataset_dir = fullfile(base_dirpath, 'raw');
scenedir = dataset_dir;
out_base_dirpath = fullfile(base_dirpath, 'processed');

% Time bin size, number of bins
num_bins = 1024; 
bin_size = 100e-9/num_bins; %approximately the bin size (in secs)
% Speed of light
c = 3e8; 
% Dark and Bright image parameter idx
dark_img_param_idx = 0; % For dark count image
psf_img_param_idx = 0; % For bright image from which the PSF/pulse wavefor is extracted
% Noise params to use during simulation
param_idx = 'all';
% 
LOW_RES = 1;  % use 0 or 1

% IMPORTANT NOTE: NOT ALL SCENES HAVE THE DIMENSIONS BELOW, IN THE INNER
% LOOP WE MODIFY THE nr and nc VARIABLES. We just create them here, to
% create a dataset name similar to the one created in the train set.
nr = 576; nc = 704;
if LOW_RES
    lres_factor = 1/8;
else
    lres_factor = 1;
end
nr = nr * lres_factor; nc = nc * lres_factor;

% Create output directory
sim_param_str = ComposeSimParamString(nr, nc, num_bins, bin_size, dark_img_param_idx, psf_img_param_idx);
outdir = fullfile(out_base_dirpath, sprintf('SimSPADDataset_%s', sim_param_str));
if ~exist(outdir, 'dir')
    mkdir(outdir)
end

% Get all scene names
scenes = GetFolderNamesInDir(dataset_dir);

% this is the 9 typical noise levels
simulation_params_T = [10 2;
                     5 2;                     
                     2 2;
                     10 10;
                     5 10;
                     2 10;
                     10 50;
                     5 50;
                     2 50];
        
% this is the extra 3 low SBR noise levels
simulation_params_E = [3 100;
                     2 100;
                     1 100];

% Extra high SNR noise levels
simulation_params_highSNR = [50, 2;
                            50, 5;
                            50, 10       
]

simulation_params_highFlux = [10, 200;
                              10, 500;
                              10, 1000       
]


% Select the simulation params to use
% for test, use 9typical or 3 extra low SBR noise levels
simulation_params = simulation_params_T;
simulation_params = simulation_params_highSNR;
simulation_params = simulation_params_highFlux;


t_s = tic;
for ss = 1:length(scenes)
    fprintf('Processing scene %s...\n',scenes{ss});

    fid = fopen(fullfile(scenedir, scenes{ss}, 'dmin.txt'));
    dmin = textscan(fid,'%f');
    dmin = dmin{1};
    fclose(fid);

    f = 3740;
    b = .1600;
    disparity = (single(imread(fullfile(scenedir, scenes{ss}, 'disp1.png'))) + dmin);
    depth = f*b ./ disparity;    
    intensity = rgb2gray(im2double(imread(fullfile(scenedir, scenes{ss}, '/view1.png'))));   

    if LOW_RES
        disp('Conduct on LR...');
        mask = disparity == dmin;
        depth(mask) = nan;

%             if ss == 7
        if isequal(scenes{ss}, 'Bowling1')
            mask = disparity == dmin;
            se = strel('disk',3);
            mask = imerode(mask,se);
            se = strel('disk',20);
            mask = imdilate(mask,se) & disparity == dmin;
            imagesc(mask);
            depth(mask) = 1.962;            
        end

        depth = full(inpaint_nans(double(depth),5));
        imagesc(depth);
        drawnow;

        r1 = 64 - mod(size(depth,1),64);
        r2 = 64 - mod(size(depth,2),64);
        r1_l = floor(r1/2);
        r1_r = ceil(r1/2);
        r2_l = floor(r2/2);
        r2_r = ceil(r2/2);

        depth = padarray(depth, [r1_l r2_l], 'replicate', 'pre');
        depth = padarray(depth, [r1_r r2_r], 'replicate', 'post');
        intensity = padarray(intensity, [r1_l r2_l], 'replicate', 'pre');
        intensity = padarray(intensity, [r1_r r2_r], 'replicate', 'post');

        depth_hr = depth;
        intensity_hr = intensity;

        depth = imresize(depth, lres_factor, 'bicubic');        
        intensity = imresize(intensity, lres_factor, 'bicubic');
        depth = max(depth, 0);
        intensity = max(intensity, 0);
    else
        depth_hr = depth;
        intensity_hr = intensity;    
    end

    albedo = intensity;
    albedo_hr = intensity_hr;

    [x,y] = meshgrid(1:size(intensity,2),1:size(intensity,1));
    x = x - size(intensity,1)/2; 
    y = y - size(intensity,2)/2;
    X = x.*depth ./ f;
    Y = y.*depth ./ f;
    dist = sqrt(X.^2 + Y.^2 + depth.^2);
    clear x1 x2 y1 y2 X1 X2 Y1 Y2;

    [x_hr,y_hr] = meshgrid(1:size(intensity_hr,2),1:size(intensity_hr,1));
    x_hr = x_hr - size(intensity_hr,1)/2; 
    y_hr = y_hr - size(intensity_hr,2)/2;
    X_hr = x_hr.*depth_hr ./ f;
    Y_hr = y_hr.*depth_hr ./ f;
    dist_hr = sqrt(X_hr.^2 + Y_hr.^2 + depth_hr.^2);

    nr = size(dist, 1); nc = size(dist, 2);
    nr_hr = size(dist_hr, 1); nc_hr = size(dist_hr, 2);
    % Load the dark count image. If dark_img_param_idx == 0 this is just zeros
    % the dark image tells use the dark count rate at each pixel in the SPAD
    dark_img = LoadAndPreprocessDarkImg(dark_img_param_idx, nr, nc);
    
    % Load the per-pixel PSF (pulse waveform on each pixel).
    % For the simulation, we will scale each pixel pulse (signal), shift it
    % vertically (background level), and shift it horizontally (depth/ToF).
    [PSF_img, psf, pulse_len] = LoadAndPreprocessBrightPSFImg(psf_img_param_idx, nr, nc, num_bins);
    psf_data_fname = sprintf('PSF_used_for_simulation_nr-%d_nc-%d.mat', nr, nc);
    psf_data_fpath = fullfile(outdir, psf_data_fname);
    save(psf_data_fpath, 'PSF_img', 'psf', 'pulse_len');

    for zz = 1:size(simulation_params,1) 
               
        % Select the mean_signal_photons and mean_background_photons
        mean_signal_photons = simulation_params(zz,1);
        mean_background_photons = simulation_params(zz,2);
        SBR = mean_signal_photons ./ mean_background_photons;
        disp(['The mean_signal_photons: ', num2str(mean_signal_photons), ', mean_background_photon: ', ...
            num2str(mean_background_photons), ', SBR: ', num2str(SBR)]);

        % Simulate the SPAD measuements at the correct resolution
        [spad, detections, rates, range_bins] = SimulateSPADMeasurement(albedo, intensity, dist, PSF_img, bin_size, num_bins, nr, nc, mean_signal_photons, mean_background_photons, dark_img, c);
	    range_bins_hr = ToF2Bins(Dist2ToF(dist_hr, c), bin_size, num_bins, false);
        % normalize the rate function to 0 to 1
	    [norm_rates, rates_offset, rates_scaling] = NormalizePhotonRates(rates);
        rates_norm_params.rates_offset = rates_offset;
        rates_norm_params.rates_scaling = rates_scaling;
        % Estimate depths with some baseline methods
        est_range_bins.est_range_bins_lmf = EstDepthBinsLogMatchedFilter(detections, PSF_img);
        est_range_bins.est_range_bins_zncc = EstDepthBinsZNCC(detections, PSF_img);
        est_range_bins.est_range_bins_argmax = EstDepthBinsArgmax(detections);

        % pad out to be divisible by 64 in spatial dimension
        spad = full(spad);
        spad = reshape(spad, size(detections));
        if ~LOW_RES
            disp('Conducting on HR...');
            r1 = 64 - mod(bins1,64);
            r2 = 64 - mod(bins2,64);
            r1_l = floor(r1/2) + 16;
            r1_r = ceil(r1/2) + 16;
            r2_l = floor(r2/2) + 16;
            r2_r = ceil(r2/2) + 16;

            spad = padarray(spad, [r1_l r2_l 0 0], 0, 'pre');
            spad = padarray(spad, [r1_r r2_r 0 0], 0, 'post');

            depth = padarray(depth, [r1_l r2_l 0 0], 0, 'pre');
            depth = padarray(depth, [r1_r r2_r 0 0], 0, 'post');

            intensity = padarray(intensity, [r1_l r2_l 0 0], 0, 'pre');
            intensity = padarray(intensity, [r1_r r2_r 0 0], 0, 'post');
        end
        H = size(spad,1); W = size(spad,2);
        spad = reshape(spad, H*W, []);
        spad = sparse(spad); 

        if(visualize_data)
            zMin = min(range_bins(:));
            zMax = max(range_bins(:));
            clf;
            subplot(2,3,1);
            imagesc(range_bins); colorbar; caxis([zMin, zMax]); title('GT Depth Bins');
            subplot(2,3,2);
            imagesc(squeeze(sum(rates,3))); colorbar; title('Total Flux');
            subplot(2,3,3);
            imagesc(squeeze(sum(detections,3))); colorbar; title('Total Meas. Photons');
            subplot(2,3,4);
            imagesc(est_range_bins.est_range_bins_lmf); colorbar; caxis([zMin, zMax]); title('LMF Est. Depth Bins');
            subplot(2,3,5);
            imagesc(est_range_bins.est_range_bins_zncc); colorbar; caxis([zMin, zMax]); title('ZNCC Est. Depth Bins');
            subplot(2,3,6);
            imagesc(est_range_bins.est_range_bins_argmax); colorbar; caxis([zMin, zMax]); title('Argmax Est Depth Bins');
            pause(0.1);
        end

        % save sparse spad detections to file
        % the 'spad' is the simulated data, 'depth' is the GT 2D depth map,
        % which is 72*88, 'depth_hr' is 576*704. 'intensity' is the
        % gray image, 72*88. 'rates' is actually the GT 3D histogram.
        out_fname = sprintf('spad_%s_%s_%s.mat', scenes{ss}, num2str(mean_signal_photons), num2str(mean_background_photons));
        out_fpath = fullfile(outdir, out_fname);
        SaveSimulatedSPADImg(out_fpath, spad, SBR, range_bins, range_bins_hr, est_range_bins, rates_norm_params, norm_rates, intensity, intensity_hr, mean_signal_photons, mean_background_photons, bin_size)

    end
end
t_cost = toc(t_s);
disp(['Time cost: ', num2str(t_cost)]);

