% Set the parameters

% Add paths
addpath('./TrainData/nyu_utils');
% Time bin size, number of bins
bin_size = 80e-12; %approximately the bin size (in secs)
num_bins = 1024; 
% Spatial resolution (vertical and horizontal
res = 64;
% Speed of light
c = 3e8; 
% Dark and Bright image parameter idx
dark_img_param_idx = 1; % For dark count image
psf_img_param_idx = 1; % For bright image from which the PSF/pulse wavefor is extracted
% Noise params to use during simulation
param_idx = 1;

% Directory where ConvertRGBD.m outputs the data
dataset_dir = './TrainData/processed';
% Directory where we will output the simulated data
sim_param_str = ComposeSimParamString(res, res, num_bins, bin_size, dark_img_param_idx, psf_img_param_idx);
output_base_dir = sprintf('./TrainData/SimSPADDataset_%s', sim_param_str);

if ~exist(output_base_dir, 'dir')
    mkdir(output_base_dir)
end
% Noise parameters to use.
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
    disp('Simulating: high flux noise levels');
    simulation_params = [100 100;
                        100 200;
                        100 800;
                        100 2000;
                        100 10000];
else
    error('Bad param_idx value');
end


