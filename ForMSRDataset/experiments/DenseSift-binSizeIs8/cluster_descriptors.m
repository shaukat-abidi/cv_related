%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%Second Part of Script%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply K-Means to cluster above accumulated descriptors
  clear all,close all,clc
  tic
  load('dsift_dataset.mat','-mat');
  dsift_load = toc
  K=400;
  data=cat(2,descriptors_bag{:});
  tic
  clear descriptors_bag
  downsampled_data = data(:,1:10:end);
  clear data
  [C] = vl_ikmeans(downsampled_data,K,'method', 'elkan');
  elkan=toc
  save('400_clusters_for_descriptors.mat','C');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%