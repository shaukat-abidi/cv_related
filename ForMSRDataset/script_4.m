clear all
close all
clc

%Adding path to functions directory
addpath('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/functions');

strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/org/';
text_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/txt/';
gt_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/GT/';

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForMSRDataset/load_files/';
load_filename_1 = strcat(save_path,'400_clusters_for_descriptors.mat');
load_filename_2 = strcat(save_path,'train_data.mat');
load_filename_3 = strcat(save_path,'labels.mat');
load_filename_4 = strcat(save_path,'to_remove.mat');%for hacky fix

%loading files
load(load_filename_1,'-mat');  
load(load_filename_2,'-mat');
load(load_filename_3,'-mat');
load(load_filename_4,'-mat');
cluster_centres = double(C);
clear C

% A very hacky fix: Some examples in trainingData have NaN and Inf
% We are removing those examples and replacing them by 
trainingData(to_remove,:)=[];
label(to_remove)=[];

new_dataset = dataset(trainingData,label);

%Learning logistic regression classifier
w=loglc(new_dataset);

save_filename_1 = strcat(save_path,'learned_weights.mat');
save(save_filename_1,'w','-v7.3');