clear all
close all

% Set input data filepaths for middlebury dataset
dataset_id = 'middlebury';
scene_name = 'spad_Moebius';
sbr_params = '10_1000';
sbr_params = '2_50';
scene_id = [scene_name, '_', sbr_params];
data_dirpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0/';
data_dirpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-144_nc-176_nt-1024_tres-98ps_dark-0_psf-0/';

% Set input data filepaths for linospad dataset
dataset_id = 'linospad';
scene_name = 'elephant';
scene_name = 'stairs_ball';
scene_name = 'lamp';
scene_name = 'stuff';
scene_id = scene_name;
data_dirpath = '2018SIGGRAPH_lindell_test_data/captured/';

% interpolation scale factor
scale_xy = 1; scale_z = 2.0;
d_min = 1.40; d_max = 2.17;

if(strcmp(scene_name, 'spad_Art'))
    d_min = 1.40; d_max = 2.17;
elseif(strcmp(scene_name, 'spad_Reindeer'))
    d_min = 1.30; d_max = 2.3;
elseif(strcmp(scene_name, 'stuff'))
    d_min = 0.6; d_max = 1.5;
    scale_xy = 1; scale_z = 1.0;
    frame_num = 1;
elseif(strcmp(scene_name, 'elephant'))
    d_min = 0.5; d_max = 1.5;
    scale_xy = 1; scale_z = 1.0;
    frame_num = 1;
elseif(strcmp(scene_name, 'lamp'))
    d_min = 0.5; d_max = 1.5;
    scale_xy = 1; scale_z = 1.0;
    frame_num = 1;
elseif(strcmp(scene_name, 'stairs_ball'))
    d_min = 0.9; d_max = 2.2;
    scale_xy = 1; scale_z = 1.0;
    frame_num = 10;
end

% load spad data
scene_fname = [scene_id, '.mat'];
spad_data_fpath = fullfile(data_dirpath, scene_fname);
spad_data = load(spad_data_fpath);
if(strcmp(dataset_id, 'middlebury'))
    [nrows, ncols, ntbins] = size(spad_data.rates);
    bin_size = spad_data.bin_size;
    hist_img = reshape(full(spad_data.spad), nrows, ncols, ntbins);
    gt_flux = spad_data.rates;
elseif(strcmp(dataset_id, 'linospad'))
    nrows = 256; ncols = 256; ntbins = 1536;
    bin_size = 26e-12;
    hist_img = reshape(full(spad_data.spad_processed_data{frame_num}), ntbins, nrows, ncols);
    hist_img = permute(hist_img, [2, 3, 1]);
%     scale_factor = 0.5;
%     hist_img = imresize3(hist_img, [128,128,1536]);
%     bin_size = (26e-12) / scale_factor;
else
    error('invalid dataset id')
end
[nrows, ncols, ntbins] = size(hist_img);

% create output directory where things will be stored
out_dirpath = fullfile('results/raw_figures/histogram_image_vis', dataset_id);
if(not(isfolder(out_dirpath)))
    mkdir(out_dirpath);
end
out_fname = [scene_id, '_', num2str(ntbins), 'x', num2str(nrows), 'x', num2str(ncols)];
out_fpath = fullfile(out_dirpath, out_fname);

%% Variables
% bin_t = 80e-12; 
bin_t = bin_size;
c = 3e8; 


start_idx = floor((2*d_min/c)/bin_t);
end_idx = ceil((2*d_max/c)/bin_t);




% intensity correction
gamma = 1.5; intensity_th = 0.15; a = 1;

% set flux 
flux_tmp = hist_img;
% % load ground-truth flux
% load('flux_map.mat')
% flux_tmp = flux_map_set;





%% Trim flux
flux_tmp(flux_tmp<0) = 0;
flux_tmp = flip(flux_tmp, 1);
flux_tmp = flip(flux_tmp, 2);
flux_tmp = flux_tmp(:, :, start_idx:end_idx);

size_y = size(flux_tmp, 1);
size_x = size(flux_tmp, 2);
size_z = size(flux_tmp, 3);



%% Interpolation
% scale along z direction
N_z = round(size_z*scale_z);
step_z = (size_z - 1)/(N_z-1);
t = 1 : size_z;
tq = 1 : step_z : size_z;

flux_z = zeros(size_y, size_x, N_z);

for y = 1 : size_y
    for x = 1 : size_x
        
        v = squeeze(flux_tmp(y, x, :));
        vq = interp1(t, v, tq);
        
        flux_z(y, x, :) = vq;
        
    end
end


% scale along x & y directions
N_x = round(size_x*scale_xy);
N_y = round(size_y*scale_xy);
flux = zeros(N_y, N_x, N_z);

for z = 1 : N_z
    
    img = flux_z(:,:,z);
    img_scaled = imresize(img, scale_xy);
    flux(:,:,z) = img_scaled;
    
end

flux(flux<0) = 0;



%% Create 3D point cloud
flux_max = prctile(flux(:), 99);


N_total = N_x*N_y*N_z;
points = zeros(N_total, 3);
rgb = zeros(N_total, 3);

idx = 1;
for y = 1 : N_y
    for x = 1 : N_x
        for z = 1 : N_z
            
            intensity = flux(y,x,z)/flux_max;
            intensity = a*intensity.^gamma;
            intensity(intensity>1) = 1;

            if intensity >= intensity_th
                
                points(idx, 1) = z;
                points(idx, 2) = x;
                points(idx, 3) = y;
                
                
                % black and white
                t_color = intensity*[1, 1, 1];
                

                rgb(idx, :) = t_color;

                idx = idx + 1;
                
            end
            
        end
    end
end



% show point cloud
ptCloud = pointCloud(points, 'Color', rgb);
figure; title(['\color{white}','']); xlabel(''); ylabel(''); zlabel('');
fig = gcf; fig.Color = 'black'; fig.InvertHardcopy = 'off';
pcshow(ptCloud, 'MarkerSize', 1e-3);
box on; ax =gca; ax.BoxStyle = 'full'; ax.XColor = 'red'; ax.YColor = 'red'; ax.ZColor = 'red'; ax.LineWidth = 4;
xticks([]); yticks([]); zticks([]);
set(gcf, 'Position', get(0, 'Screensize'));

if(strcmp(scene_name, 'spad_Reindeer'))
    view([-74.5537, 3.0547]);
elseif(strcmp(scene_name, 'spad_Moebius'))
    view([-69.9854, 5.2588]);
elseif(strcmp(scene_name, 'stuff'))
%     view([-103.3623, 5.0361])
    view([-63.8652,6.9580])
else
    view([-63.8652,6.9580])
%     view([-69, 15]);
end


saveas(gcf, out_fpath, 'svg');


% exportgraphics(gca,[out_fpath,'.png'], 'Resolution', 900);

