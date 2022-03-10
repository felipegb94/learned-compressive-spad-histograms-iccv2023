
% The directory where you extracted the raw dataset.
clear all;close all;clc
addpath(genpath('./intrinsic_texture'));
addpath('./nyu_utils');
datasetDir = './raw_corrupted';
% datasetDir = './raw_new_download';

% datasetDir = '../../raw';

% get the scene names
scenes = ls(datasetDir);
scenes = regexp(scenes, '(\s+|\n)', 'split');
scenes(end) = [];
camera_params;

t_s = tic;
% parfor ss = 1:length(scenes) % original is parfor
for ss = 1:length(scenes) % original is parfor

    sceneName = scenes{ss};
    
    disp('starting!');

    % The name of the scene to demo.
    outdir = ['./processed_test/' sceneName];

    mkdir(outdir);

    % The absolute directory of the 
    sceneDir = sprintf('%s/%s', datasetDir, sceneName);

    % Reads the list of frames.
    frameList = get_synched_frames(sceneDir);

    % Displays each pair of synchronized RGB and Depth frames.
    idx = 1 : 10 : numel(frameList);
    
    disp(outdir)

    for ii = 1:length(idx)
        fprintf('    Processing: %s_%04d\n', sceneName, ii);

        % check if already exists
        depth_out = sprintf('%s/depth_%04d.mat', outdir, idx(ii));
        albedo_out = sprintf('%s/albedo_%04d.mat', outdir, idx(ii));
        intensity_out = sprintf('%s/intensity_%04d.mat', outdir, idx(ii));
        dist_out = sprintf('%s/dist_%04d.mat',outdir, idx(ii));
        dist_out_hr = sprintf('%s/dist_hr_%04d.mat',outdir, idx(ii));
% 
%         if exist(depth_out,'file') && exist(albedo_out,'file') ...
%                 && exist(intensity_out,'file') && exist(dist_out,'file') ...
%                 && exist(dist_out_hr,'file')
%                 disp('continuing');
%                 continue;
%         end
        if  exist(albedo_out,'file') ...
                && exist(intensity_out,'file') && exist(dist_out,'file') ...
                && exist(dist_out_hr,'file')
                disp('continuing');
                continue;
        end

%         if(strcmp(sceneName, 'living_room_0059'))
%             if((ii == 195) || (ii == 204) || (ii == 208))
%                 fprintf('skipping living_room_0059_%04d for now\n', ii)
%                 continue;
%             end
%         elseif(strcmp(sceneName, 'living_room_0057'))
%             if((ii == 123))
%                 fprintf('skipping living_room_0057_%04d for now\n', ii)
%                 continue;
%             end
%         elseif(strcmp(sceneName, 'office_0001c'))
%             if((ii == 83))
%                 fprintf('skipping office_0001c_%04d for now\n', ii)
%                 continue;
%             end
%         end
        
        try
            imgRgb = imread([sceneDir '/' frameList(idx(ii)).rawRgbFilename]);
            imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(idx(ii)).rawDepthFilename]));

            % Crop the images to include the areas where we have depth information.
            imgRgb = crop_image(imgRgb);
            imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
            imgDepthAbs = crop_image(imgDepthProj);
            imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));
          
            % get distance from the depth image
            cx = cx_d - 41 + 1;
            cy = cy_d - 45 + 1;
            [xx,yy] = meshgrid(1:561, 1:427);
            X = (xx - cx) .* imgDepthFilled / fx_d;
            Y = (yy - cy) .* imgDepthFilled / fy_d;
            Z = imgDepthFilled;
            imgDist_hr = sqrt(X.^2 + Y.^2 + Z.^2);
           
            % estimate the albedo image and save the outputs
            I = im2double(imgRgb);
            I = imresize(I, [512, 512], 'bilinear');
            imgDepthFilled = imresize(imgDepthFilled, [512,512], 'bilinear');
            imgDist = imresize(imgDist_hr, [256,256], 'bilinear');
            imgDist_hr = imresize(imgDist_hr, [512,512], 'bilinear');
            S = RollingGuidanceFilter(I, 3, 0.1, 4);
            [albedo, ~] = intrinsic_decomp(I, S, imgDepthFilled, 0.0001, 0.8, 0.5);
            intensity = rgb2gray(I);

            dist = imgDist;
            intensity = im2uint8(intensity);
            dist_hr = imgDist_hr;
            ConvertRGBDParsave(albedo_out, dist_out, intensity_out, dist_out_hr, albedo, dist, intensity, dist_hr)
             
        catch e
            fprintf(1,'ERROR: %s\n',e.identifier);
            fprintf(1,'%s',e.message);
            continue;
        end
    end
end
t_cost = toc(t_s);
disp(['Time spend: ', num2str(t_cost/3600/24), ' days']);
