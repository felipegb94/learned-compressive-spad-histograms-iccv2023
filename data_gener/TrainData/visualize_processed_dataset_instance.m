clear;

sceneID = 'home_office_0002';
captureID = '0861';
sceneID = 'study_room_0003';
captureID = '0011';

dirpath = sprintf('./processed/%s/', sceneID);


% Load each image associated with the above scene
albedo = load(sprintf('./processed/%s/albedo_%s.mat', sceneID, captureID)).albedo;
intensity = load(sprintf('./processed/%s/intensity_%s.mat', sceneID, captureID)).intensity;
dist = load(sprintf('./processed/%s/dist_%s.mat', sceneID, captureID)).dist;
dist_hr = load(sprintf('./processed/%s/dist_hr_%s.mat', sceneID, captureID)).dist_hr;


clf;
subplot(2,2,1); imshow(albedo); title(['albedo: ', num2str(size(albedo))])
subplot(2,2,2); imshow(intensity); title(['intensity: ', num2str(size(intensity))])
subplot(2,2,3); imagesc(dist); title(['dist: ', num2str(size(dist))]);colorbar; colormap(gca, 'pink')
subplot(2,2,4); imagesc(dist_hr); title(['dist\_hr: ', num2str(size(dist_hr))]);colorbar; colormap(gca, 'pink')



sgtitle([sceneID, '\_', captureID]);