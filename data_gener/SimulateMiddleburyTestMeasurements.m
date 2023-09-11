clear; close all;

visualize_data = true;
% Add paths
load('/nobackup/bhavya/datasets/sunrgbd/SUNRGBDMeta3DBB_v2.mat');
load('/nobackup/bhavya/datasets/sunrgbd/SUNRGBDMeta2DBB_v2.mat');
addpath('/nobackup/bhavya/votenet/sunrgbd/sunrgbd_trainval/')
% addpath('./TestData/middlebury/nyu_utils');

% Set paths
% base_dirpath = './TestData/middlebury';
base_dirpath = '/nobackup/bhavya/votenet/sunrgbd/sunrgbd_trainval/';
% dataset_dir = fullfile(base_dirpath, 'raw');
dataset_dir = base_dirpath;
scenedir = fullfile(base_dirpath, 'image');
depthdir = fullfile(base_dirpath, 'depth');
out_base_dirpath = fullfile(base_dirpath, 'processed');

% Speed of light
c = 3e8; 
% Time bin size, number of bins
num_bins = 1024; 
repetition_period = 600e-9;
unambiguous_depth_range = ToF2Dist(repetition_period, c);
bin_size = repetition_period/num_bins; %approximately the bin size (in secs)
% Dark and Bright image parameter idx
dark_img_param_idx = 0; % For dark count image
psf_img_param_idx = 0; % For bright image from which the PSF/pulse wavefor is extracted (see LoadAndPreprocessBightPSFImg.m for options)
% Noise params to use during simulation
param_idx = 'all';
% 
LOW_RES = 1;  % use 0 or 1
% additional identifier for dataset
dataset_sim_id = 'SimSPADDataset'; % regular test set
% dataset_sim_id = 'MaskedHighTimeBinsSimSPADDataset'; % Test set with the time bins above a certain threshold being set to 0
% dataset_sim_id = 'LargeDepthSimSPADDataset';
% dataset_sim_id = 'LowSBRSimSPADDataset'; % low sbr test set 
% dataset_sim_id = 'HighSignalSimSPADDataset'; % low signal test set


% IMPORTANT NOTE: NOT ALL SCENES HAVE THE DIMENSIONS BELOW, IN THE INNER
% LOOP WE MODIFY THE nr and nc VARIABLES. We just create them here, to
% create a dataset name similar to the one created in the train set.
nr = 576; nc = 704;
if LOW_RES
    lres_factor = 1/8;
%     lres_factor = 1/4;
%     lres_factor = 1/2;
else
    lres_factor = 1;
end
nr = nr * lres_factor; nc = nc * lres_factor;

% Create output directory
sim_param_str = ComposeSimParamString(nr, nc, num_bins, bin_size, dark_img_param_idx, psf_img_param_idx);
outdir = fullfile(out_base_dirpath, sprintf('%s_%s', dataset_sim_id,sim_param_str));
if ~exist(outdir, 'dir')
    mkdir(outdir)
end

% Get all scene names
% scenes = GetFolderNamesInDir(dataset_dir);
scenes = load(fullfile(base_dirpath, 'train_data_idx.txt'));
for ss = 1:length(scenes)
    aa = dlmread(fullfile("/nobackup/bhavya/datasets/sunrgbd", SUNRGBDMeta(scenes(ss)).sequenceName, 'intrinsics.txt'));
    focallength(ss) = aa(1,1);
    % TODO: size for each image is different, we should use original size for inference.
    % color_image = imread(fullfile(scenedir, sprintf('%s.jpg', scenes{ss})))
    % dims(ss) = size(color_image);
end
scenes = num2str(scenes, '%06d');
scenes = string(scenes);


if(strcmp(dataset_sim_id, 'LowSBRSimSPADDataset'))
    simulation_params_lowSBR = [3 100;
                                2 100;
                                1 100;
                                2 50;
                                10, 500;
                                10, 1000;
                                50, 5000;
                                100, 5000;
                                100, 10000;
                                100, 20000
    ];
    simulation_params = simulation_params_lowSBR;
elseif(strcmp(dataset_sim_id, 'HighSignalSimSPADDataset'))
    simulation_params_lowSBR = [
                                200, 500;
                                200, 2000;
                                200, 5000;
                                200, 10000;
                                200, 20000
    ];
    simulation_params = simulation_params_lowSBR;
else
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
            
%     % this is the extra 3 low SBR noise levels
%     simulation_params_E = [3 100;
%                          2 100;
%                          1 100];
        
    simulation_params_highFlux = [10, 200;
                                  10, 500;
                                  10, 1000
    ];
    
    
    simulation_params_highFlux_medSNR = [ 50, 50; 
                                  50, 200;
                                  50, 500;
                                  50, 1000
    ];

    simulation_params = [simulation_params_T; simulation_params_highFlux; simulation_params_highFlux_medSNR];
    
end
% 

% Sensor parameters from https://github.com/facebookresearch/omnivore/issues/12
% don't need these values for now.
% datasets = ["kv1", "kv1_b", "kv2", "realsense", "xtion"];
% baselines = [0.075, 0.075, 0.075, 0.095, 0.095];
% sensor_to_params = dictionary(datasets, baselines)


t_s = tic;
for ss = 1:10
    % length(scenes)
    fprintf('Processing scene %s...\n',scenes{ss});

    % fid = fopen(fullfile(scenedir, scenes{ss}, 'dmin.txt'));
    % dmin = textscan(fid,'%f');
    dmin = 0.;
    % dmin{1};
    % fclose(fid);

    % f = 3740;
    f = focallength(ss);
    % b = .1600;
    % disparity = (single(imread(fullfile(depthdir, sprintf('%s.png', scenes{ss})) ) ) + dmin);
    % depth = f*b ./ disparity;    
    depth = (single(imread(fullfile(depthdir, sprintf('%s.png', scenes{ss})) ) ) + dmin);
    disparity = depth; % This is not disparity, just naming the variable that to keep the code same as before.
    depth = depth./1000;
    % depth(depth>50)=50;
    % depth(depth<0.01)=0.01;
    % intensity = rgb2gray(im2double(imread(fullfile(scenedir, scenes{ss}, '/view1.png'))));   
    intensity = rgb2gray(im2double(imread(fullfile(scenedir, sprintf('%s.jpg', scenes{ss})))));   

    % When simulating the large depth dataset apply a fixed offset
    if(strcmp(dataset_sim_id, 'LargeDepthSimSPADDataset'))
        depth_offset = 7;
        depth = depth + depth_offset;
    end

    max_scene_depth = max(depth(:));
    min_scene_depth = min(depth(:));
    fprintf('    Max scene depth: %f...\n',max_scene_depth);
    fprintf('    Min scene depth: %f...\n',min_scene_depth);
    size(depth)

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
        
        % Zero out time bins above a certain distance
        if(strcmp(dataset_sim_id, 'MaskedHighTimeBinsSimSPADDataset'))
            depth_thresh = 9;
            tof_thresh = Dist2ToF(depth_thresh, c);
            bin_thresh = ToF2Bins(tof_thresh, bin_size, num_bins, true);
            % mask bins above bin_thresh
            detections(:,:,bin_thresh:end) = 0;
            spad = sparse(reshape(detections, nr*nc, []));
        end

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
            r1 = 64 - mod(size(spad,1),64);
            r2 = 64 - mod(size(spad,2),64);
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
        SaveSimulatedSPADImgSmall(out_fpath, detections, SBR, range_bins, intensity, bin_size, f)

    end
end
t_cost = toc(t_s);
disp(['Time cost: ', num2str(t_cost)]);

