function SimulateSUNRGBDMeasurements(startidx, endidx) 
	%clear; close all;
	visualize_data = false;
	% Add paths
	% load('/nobackup/bhavya/datasets/sunrgbd/SUNRGBDMeta3DBB_v2.mat');
	% load('/nobackup/bhavya/datasets/sunrgbd/SUNRGBDMeta2DBB_v2.mat');
	%addpath('/nobackup/bhavya/votenet/sunrgbd/sunrgbd_trainval/')
	
	% Set paths
	base_dirpath = '/srv/home/bgoyal2/Documents/votnet/sunrgbd/sunrgbd_trainval/';
	dataset_dir = base_dirpath;
	scenedir = fullfile(base_dirpath, 'image');
	depthdir = fullfile(base_dirpath, 'depth');
	out_base_dirpath = fullfile(base_dirpath, 'processed_full_lowflux');
	
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
	% additional identifier for dataset
	dataset_sim_id = 'SimSPADDataset'; % regular test set
	
	
	% IMPORTANT NOTE: NOT ALL SCENES HAVE THE DIMENSIONS BELOW, IN THE INNER
	% LOOP WE MODIFY THE nr and nc VARIABLES. We just create them here, to
	% create a dataset name similar to the one created in the train set.
	nr = 576; nc = 704;
	lres_factor = 1;
	nr = nr * lres_factor; nc = nc * lres_factor;
	
	% Create output directory
	sim_param_str = ComposeSimParamString(nr, nc, num_bins, bin_size, dark_img_param_idx, psf_img_param_idx);
	outdir = fullfile(out_base_dirpath, sprintf('%s_%s', dataset_sim_id,sim_param_str));
	if ~exist(outdir, 'dir')
	    mkdir(outdir)
	end
	
	% Get all scene names
	scenes = load(fullfile(base_dirpath, 'val_data_idx.txt'));
	% for ss = 1:length(scenes)
	%     aa = dlmread(fullfile("/nobackup/bhavya/datasets/sunrgbd", SUNRGBDMeta(scenes(ss)).sequenceName, 'intrinsics.txt'));
	%     focallength(ss) = aa(1,1);
	%     % TODO: size for each image is different, we should use original size for inference.
	%     % color_image = imread(fullfile(scenedir, sprintf('%s.jpg', scenes{ss})))
	%     % dims(ss) = size(color_image);
	% end
	scenes = num2str(scenes, '%06d');
	scenes = string(scenes);

	for ss = 1:length(scenes)
	    K = dlmread(fullfile(base_dirpath, 'calib', sprintf('%s.txt', scenes{ss} )));
	    allK{ss} = K;
	    fx(ss) = K(2, 1); fy(ss) = K(2, 5);
	    cx(ss) = K(2, 7); cy(ss) = K(2, 8);
	end
	
	simulation_params_highFlux_medSNR = [ 5, 1; 
	                                   5, 5;
	                                  5, 10;
	                                   5, 50
	];
	simulation_params = [simulation_params_highFlux_medSNR];
	
	% 
	% Sensor parameters from https://github.com/facebookresearch/omnivore/issues/12
	% don't need these values for now.
	% datasets = ["kv1", "kv1_b", "kv2", "realsense", "xtion"];
	% baselines = [0.075, 0.075, 0.075, 0.095, 0.095];
	% sensor_to_params = dictionary(datasets, baselines)
	
	startidx = max(startidx, 1);
	endidx = min(endidx, length(scenes));
	t_s = tic;
	for ss = startidx:endidx
	    fprintf('Processing scene %s...\n',scenes{ss});
	    out_fname = sprintf('spad_%s_%s_%s.mat', scenes{ss}, num2str(simulation_params(end,1)), num2str(simulation_params(end,2)));
		out_fpath = fullfile(outdir, out_fname)
	    if exist(out_fpath)
		    continue
	    end
	
	    dmin = 0.;
	    % f = focallength(ss);
	    depth = (single(imread(fullfile(depthdir, sprintf('%s.png', scenes{ss})) ) ) + dmin);
	    depth = depth./1000;
	    % No need to clip depth maps for now
	    % depth(depth>50)=50;
	    % depth(depth<0.01)=0.01;
	    intensity = rgb2gray(im2double(imread(fullfile(scenedir, sprintf('%s.jpg', scenes{ss})))));   
	
	    max_scene_depth = max(depth(:));
	    min_scene_depth = min(depth(:));
	    fprintf('    Max scene depth: %f...\n',max_scene_depth);
	    fprintf('    Min scene depth: %f...\n',min_scene_depth);
	    size(depth)
	
	    mask = depth == dmin;
	    depth(mask) = nan;
	
	    depth = full(inpaint_nans(double(depth),5));
	    imagesc(depth);
	    drawnow;
	
	    depth = max(depth, 0);
	    intensity = max(intensity, 0);
	
	    albedo = intensity;
	
	    [x,y] = meshgrid(1:size(intensity,2),1:size(intensity,1));
	    % x = x - size(intensity,2)/2; 
	    % y = y - size(intensity,1)/2;
	    % X = x.*depth ./ f;
	    % Y = y.*depth ./ f;
	    X = (x - cx(ss)).*depth ./ fx(ss);
	    Y = (y - cy(ss)).*depth ./ fy(ss);
	    dist = sqrt(X.^2 + Y.^2 + depth.^2);
	    clear x1 x2 y1 y2 X1 X2 Y1 Y2;
	
	    nr = size(dist, 1); nc = size(dist, 2);
	    % Load the dark count image. If dark_img_param_idx == 0 this is just zeros
	    % the dark image tells use the dark count rate at each pixel in the SPAD
	    dark_img = LoadAndPreprocessDarkImg(dark_img_param_idx, nr, nc);
	    
	    % Load the per-pixel PSF (pulse waveform on each pixel).
	    % For the simulation, we will scale each pixel pulse (signal), shift it
	    % vertically (background level), and shift it horizontally (depth/ToF).
	    [PSF_img, psf, pulse_len] = LoadAndPreprocessBrightPSFImg(psf_img_param_idx, nr, nc, num_bins);
	    psf_data_fname = sprintf('PSF_used_for_simulation_nr-%d_nc-%d.mat', nr, nc);
	    psf_data_fpath = fullfile(outdir, psf_data_fname);
	    if ~exist(psf_data_fpath)
	    	save(psf_data_fpath, 'PSF_img', 'psf', 'pulse_len');
	    end
	
	    for zz = 1:size(simulation_params,1) 
	               
	        % Select the mean_signal_photons and mean_background_photons
	        mean_signal_photons = simulation_params(zz,1);
	        mean_background_photons = simulation_params(zz,2);
	        SBR = mean_signal_photons ./ mean_background_photons;
	        disp(['The mean_signal_photons: ', num2str(mean_signal_photons), ', mean_background_photon: ', ...
	            num2str(mean_background_photons), ', SBR: ', num2str(SBR)]);
	
	        % Simulate the SPAD measuements at the correct resolution
	        [spad, detections, rates, range_bins] = SimulateSPADMeasurement(albedo, intensity, dist, PSF_img, bin_size, num_bins, nr, nc, mean_signal_photons, mean_background_photons, dark_img, c);
	        
	        % normalize the rate function to 0 to 1
		    [norm_rates, rates_offset, rates_scaling] = NormalizePhotonRates(rates);
	        rates_norm_params.rates_offset = rates_offset;
	        rates_norm_params.rates_scaling = rates_scaling;
	        
	        % Estimate depths with some baseline methods
	        est_range_bins.est_range_bins_lmf = EstDepthBinsLogMatchedFilter(detections, PSF_img);
	        est_range_bins.est_range_bins_zncc = EstDepthBinsZNCC(detections, PSF_img);
	        est_range_bins.est_range_bins_argmax = EstDepthBinsArgmax(detections);
	
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
	        SaveSimulatedSPADImgSmall(out_fpath, spad, SBR, range_bins, intensity, bin_size, allK{ss}, num_bins)
	
	    end
	end
	t_cost = toc(t_s);
	disp(['Time cost: ', num2str(t_cost)]);
end
