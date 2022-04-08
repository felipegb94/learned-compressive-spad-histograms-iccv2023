function [folder_names] = GetFolderNamesInDir(dirpath)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
% get the scene names

d = dir(dirpath);
% remove all files (isdir property is 0)
dfolders = d([d(:).isdir]);
% remove '.' and '..' 
dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));


folder_names = {dfolders(:).name};

% scenes = ls(dirpath);
% tmp = cell(size(scenes,1),1);
% if ispc
%     % If we are on windows machine generate fnames differently
%     scenes = strtrim(string(scenes));
% else
%     scenes = regexp(scenes, '(\s+|\n)', 'split');
%     scenes(end) = [];
% end

end