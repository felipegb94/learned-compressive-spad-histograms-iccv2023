function SimulateMeasurements(param_idx)
% Simulate train dataset with RGB-D + albedo generated by the ConvertRGBD.m.
% param_idx - Specify which signal/background level to generate.
% 1: the 9 typical noise levels;
% 2: the 9 typical noise levels + 3 low SBR noise levels;

    addpath('./nyu_utils');
    dataset_dir = './processed';
    
    bin_size = 80e-12; %approximately the bin size 
    num_bins = 1024; 
    res = 64;
    c = 3e8;

    if param_idx == 1
        disp('Simulating: 9 typical noise levels');
        simulation_params = [10 2;
                         5 2;
                         2 2;
                         10 10; 
                         5 10; 
                         2 10; 
                         10 50; 
                         5 50; 
                         2 50];
    elseif param_idx == 2
        fprintf('Simulating: 9 typical noise levels + 3 low SBR levels');
        simulation_params = [10 2;
                         5 2;
                         2 2;
                         10 10; 
                         5 10; 
                         2 10; 
                         10 50; 
                         5 50; 
                         2 50;
                         3 100;
                         2 100;
                         1 100];
    elseif param_idx == 3
        fprintf('Simulating: 9 typical noise levels + 3 low SBR levels');
        simulation_params = [10 2];
    else
        error('Bad param_idx value');
    end

    % load in psf image and dark image
    load 'bright_img.mat';
    load 'dark_img.mat';

    % process dark counts / psf
    dark_img = repmat(mean(dark_img(:,1:res)),[res,1])';

    % isolate psf
    bright_img = bright_img(:,1:res);
    [~,idx] = max(bright_img,[],1);
    for ii = 1:res
       tmp = circshift(bright_img(:,ii),10 - idx(ii));
       psf(:,ii) = flipud(tmp(1:16));
    end
    psf = psf ./ sum(psf,1);
    psf(isnan(psf)) = 0;

    % get the scene names
    scenes = ls(dataset_dir);
    tmp = cell(size(scenes,1),1);
    if ispc
        scenes = strtrim(string(scenes));
    else
        scenes = regexp(scenes, '(\s+|\n)', 'split');
        scenes(end) = [];
    end
    
    for ss = 1:length(scenes)
        scene_name=scenes{ss};
        fprintf('Processing scene: %s\n', scene_name); 

        % The name of the scene to demo.
        outdir = fullfile(dataset_dir,'/', scene_name);
        if ~exist(outdir, 'dir')
            mkdir(outdir)
        end
        
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
            spad_out = sprintf('%s/spad_%s_p%d.mat', outdir, nums{ii}, param_idx);
            % Skip file if it already exists
            if exist(spad_out,'file')
                fprintf("Continuiing. %s already exists \n", spad_out);
                continue;
            end
            
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

            % check for valid albedo
            if any(isnan(albedo(:)))
               fprintf('Found nan albedo\n');
               continue;  parsave
            end
            
            % if necessary, inpaint depth values
            if any(dist_hr(:) == 0)
                se = strel('disk',3, 0);
                mask = dist_hr == 0;
                mask = imdilate(mask, se);
                dist_hr(mask) = nan;
                dist_hr = full(inpaint_nans(dist_hr));
            end
            albedo(albedo<0) = 0;  
            intensity(intensity<0) = 0;
            albedo_hr = albedo;
            
            % resize albedo and intensity to 64 x 64
            albedo = imresize(albedo, [res res], 'bilinear');
            intensity = imresize(intensity, [res res], 'bilinear');
            dist = imresize(dist_hr, [res res], 'bilinear');
            
            % convert to time of flight
            d = dist;
            tof = dist * 2 / c;
            tof_hr = dist_hr * 2 / c; 

            % convert to bin number
            range_bins = round(tof ./ bin_size);
            range_bins_hr = tof_hr ./ bin_size;
            if any(reshape(range_bins > num_bins, 1, []))
                fprintf('some photon events out of range\n');
            end
            range_bins = min(range_bins, num_bins);   
            range_bins = max(range_bins, 1);
            range_bins_hr = min(range_bins_hr, num_bins); 
            range_bins_hr = max(range_bins_hr, 1);

            % set a number of signal photons per pixel
            alpha = albedo .* 1./ dist.^2;
            
            if param_idx ==1
                param_tmp = simulation_params(randi(9), :);
                mean_signal_photons = param_tmp(1);
                mean_background_photons = param_tmp(2);
                SBR = mean_signal_photons / mean_background_photons;
                disp(['Selecting signal photons: ', num2str(mean_signal_photons), ' background photons: ', num2str(mean_background_photons), ' SBR: ', num2str(SBR)]);
            elseif param_idx ==2
                param_tmp = simulation_params(randi(12), :);
                mean_signal_photons = param_tmp(1);
                mean_background_photons = param_tmp(2);
                SBR = mean_signal_photons / mean_background_photons;
                disp(['Selecting signal photons: ', num2str(mean_signal_photons), ' background photons: ', num2str(mean_background_photons), ' SBR: ', num2str(SBR)]);
            end
            
            % add albedo/range/lighting/dark count effects
            signal_ppp = alpha ./ mean(alpha(:)) .* mean_signal_photons;
            % make approximately correct ratio between ambient light and dark count
            ambient_ppp = dark_img + mean_background_photons .* intensity ./ mean(intensity(:));
            % apply a global scale to both to get the exact desired sbr
            ambient_ppp = ambient_ppp ./ mean(ambient_ppp(:)) .* mean_background_photons;
            
            % construct the inhomogeneous poisson process
            rates = zeros(res,res,num_bins);
            pulse = repmat(psf,[1,1,res]);       
            pulse = permute(pulse,[2,3,1]);
            rates(:,:,1:size(pulse,3)) = pulse;
            rates(:,:,1:size(pulse,3)) = rates(:,:,1:size(pulse,3)).*repmat(signal_ppp,[1,1,size(pulse,3)]);
            rates = rates + repmat(ambient_ppp./num_bins,[1,1,num_bins]);        
            
           % find amount to circshift the rate function
            [~, pulse_max_idx] = max(psf(:,1));
            circ_amount = range_bins - pulse_max_idx;
            for jj = 1:res
                for kk = 1:res
                    rates(jj,kk,:) = circshift(squeeze(rates(jj,kk,:)), circ_amount(jj,kk));
                end
            end
                     
            % sample the process
            detections = poissrnd(rates);
            detections = reshape(detections, res*res, []);
            spad = sparse(detections);

            % normalize the rate function to 0 to 1
            rates = (rates - min(rates,[],3))./ (max(rates,[],3) - min(rates,[],3));
            if any(isnan(detections(:))) || any(isnan(rates(:)))
                warning('NAN!');
                continue;
            end
            
            % save sparse spad detections to file
            parsave(spad_out, spad, SBR, range_bins, range_bins_hr, mean_signal_photons, rates);
       end
    end
end
