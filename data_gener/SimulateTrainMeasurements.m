%% Simulates the SPAD training data from the processed nyuv2 images
% Modified from Lindell et al. 2018, to increase code re-use between
% train/test simulation code, and also adapt the laser pulse waveform used to generate the dataset.
% The core simulation code did not change significantly, it was just
% re-organized.

% Simulates the SPAD training data from the processed nyuv2 images.
clear;

% Set the same simulation parameters used in Lindell et al and Peng et al
SetSimulateTrainParams
nr = res;
nc = res;

% Load the dark count image. If dark_img_param_idx == 0 this is just zeros
% the dark image tells use the dark count rate at each pixel in the SPAD
dark_img = LoadAndPreprocessDarkImg(dark_img_param_idx, res, res);
print_vector_stats(dark_img, "Dark Img")

% Load the per-pixel PSF (pulse waveform on each pixel).
% For the simulation, we will scale each pixel pulse (signal), shift it
% vertically (background level), and shift it horizontally (depth/ToF).
[PSF_img, psf, pulse_len] = LoadAndPreprocessBrightPSFImg(psf_img_param_idx, res, res, num_bins);
print_vector_stats(psf, "PSF")
print_vector_stats(PSF_img, "PSF Img")
psf_data_fname = 'PSF_used_for_simulation.mat';
psf_data_fpath = fullfile(output_base_dir, psf_data_fname);
save(psf_data_fpath, 'PSF_img', 'psf', 'pulse_len');


% get the scene names
scenes = ls(dataset_dir);
tmp = cell(size(scenes,1),1);
if ispc
    % If we are on windows machine generate fnames differently
    scenes = strtrim(string(scenes));
else
    scenes = regexp(scenes, '(\s+|\n)', 'split');
    scenes(end) = [];
end

% For testing
% scenes = scenes(1:2);
scenes = scenes(1:5);
scenes = scenes(1:2);

fprintf('** Simulating dataset: %s *****\n', output_base_dir);
fprintf('***********\n'); 


for ss = 1:length(scenes)
    scene_name=scenes{ss};
    fprintf('******PROCESSING SCENE: %s *****\n', scene_name); 

    % The name of the scene to demo.
    outdir = fullfile(output_base_dir,'/', scene_name);
    if ~exist(outdir, 'dir')
        mkdir(outdir)
    end
    
    % Find the images to load
    processed_files = ls([dataset_dir '/' scene_name]);
    if ispc
        dist_imgs = regexp(reshape(processed_files',1,[]), 'dist_hr_\d\d\d\d.mat', 'match');
    else
        dist_imgs = regexp(processed_files, 'dist_hr_\d\d\d\d.mat', 'match');
    end
    dist_imgs = sort(dist_imgs);
    nums = regexp(dist_imgs,'\d\d\d\d','match');
    nums = [nums{:}];

    % Displays each pair of synchronized RGB and Depth frames.
    for ii = 1 : 1 : numel(dist_imgs)
        % Output fname, keep track of the noise param_idx used
        spad_out_fname = sprintf('spad_%s_p%d.mat', nums{ii}, param_idx);
        spad_out = fullfile(outdir, spad_out_fname);
        %         spad_out = sprintf('%s/spad_%s_p%d.mat', outdir, nums{ii}, param_idx);
        % Skip file if it already exists
        if exist(spad_out,'file')
            fprintf("Continuiing. %s already exists \n", spad_out);
            continue;
        end
        % Load other files needed for simulation
        try        
            dist_hr_mat = load(sprintf('%s/%s/%s',dataset_dir, scene_name, dist_imgs{ii}));
            albedo_mat = load(sprintf('%s/%s/%s%s%s',dataset_dir, scene_name, 'albedo_', nums{ii}, '.mat'));
            intensity_mat = load(sprintf('%s/%s/%s%s%s',dataset_dir, scene_name, 'intensity_', nums{ii}, '.mat'));  
            dist_hr = dist_hr_mat.dist_hr;
            albedo = albedo_mat.albedo;
            intensity = intensity_mat.intensity;            
            albedo = squeeze(albedo(:,:,3)); % blue channel only
            intensity = im2double(intensity);
        catch e %e is an MException struct
            fprintf(1,'The identifier was:%s\n',e.identifier);
            fprintf(1,'There was an error! The message was:%s\n',e.message);
            fprintf('error loading file for %s/%s/%s\n',dataset_dir, scene_name, dist_imgs{ii});
            continue;
        end

        % check for valid albedo, if invalid skip the file
        if any(isnan(albedo(:)))
           fprintf('Found nan albedo\n');
           continue; 
        end
        % if necessary, inpaint depth values
        if any(dist_hr(:) == 0)
            se = strel('disk',3, 0);
            mask = dist_hr == 0;
            mask = imdilate(mask, se);
            dist_hr(mask) = nan;
            dist_hr = full(inpaint_nans(dist_hr));
        end
        clf; imshow(dist_hr / max(dist_hr(:)));
        % Set to 0 any negative numbers 
        albedo(albedo<0) = 0;  
        intensity(intensity<0) = 0;
        albedo_hr = albedo;
        intensity_hr = intensity;

        % randomly select the signal and background parameters from the
        % simulation params options
        rand_sim_param_idx = randi(size(simulation_params,1));
        param_tmp = simulation_params(rand_sim_param_idx, :);
        mean_signal_photons = param_tmp(1);
        mean_background_photons = param_tmp(2);
        SBR = mean_signal_photons / mean_background_photons;
        disp(['Selecting signal photons: ', num2str(mean_signal_photons), ' background photons: ', num2str(mean_background_photons), ' SBR: ', num2str(SBR)]);

        % Simulate the SPAD measuements at the correct resolution
        [spad, detections, rates, norm_rates, range_bins, range_bins_hr] = SimulateSPADMeasurement(albedo_hr, intensity_hr, dist_hr, PSF_img, bin_size, num_bins, nr, nc, mean_signal_photons, mean_background_photons, dark_img, c);

        % Check if nan and skip if so
        if any(isnan(detections(:))) || any(isnan(rates(:)))
            warning('NAN!');
            continue;
        end
            
        % save sparse spad detections to file
        SaveSimulatedSPADImg(spad_out, spad, SBR, range_bins, range_bins_hr, rates, norm_rates, mean_signal_photons, mean_background_photons, bin_size);
%         parsave(spad_out, spad, SBR, range_bins, range_bins_hr, mean_signal_photons, rates);
        
    end
end

