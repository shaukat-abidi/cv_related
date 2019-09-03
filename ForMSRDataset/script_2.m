%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%Second Part of Script%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply K-Means to cluster above accumulated descriptors
  clear all
  close all
  save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForMSRDataset/load_files/';
  load_filename_1 = strcat(save_path,'dsift_dataset.mat');

  
  %loading files
  load(load_filename_1,'-mat');
  
  K=400;
  data=cat(2,descriptors_bag{:});
  clear descriptors_bag
  downsampled_data = data(:,1:5:end);
  clear data
  tic
  [C] = vl_ikmeans(downsampled_data,K,'method', 'elkan');
  kmeans_done=toc
  %%[C] = vl_ikmeans(data,K,'algorithm', 'elkan','Initialization','plusplus');
  
  
  save_filename_1 = strcat(save_path,'400_clusters_for_descriptors.mat');
  save(save_filename_1,'C','-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%