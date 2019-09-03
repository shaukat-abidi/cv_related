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
load_filename_5 = strcat(save_path,'learned_weights.mat');


%loading files
load(load_filename_1,'-mat');  
load(load_filename_2,'-mat');
load(load_filename_3,'-mat');
load(load_filename_4,'-mat');
load(load_filename_5,'-mat');
cluster_centres = double(C);
clear C

% A very hacky fix: Some examples in trainingData have NaN and Inf
% We are removing those examples and replacing them by 
trainingData(to_remove,:)=[];
label(to_remove)=[];

new_dataset = dataset(trainingData,label);

mappedD = new_dataset*w;
predicted_labels = mappedD*labeld;

correct_labels = (predicted_labels == label);
for_inspection = [label predicted_labels correct_labels];
inspect_predictedDataset = mappedD.data;
accuracy = sum(correct_labels)/length(label) * 100;
fprintf('total labels = %d Correctly classified = %d Accuracy=%d \n',length(label),sum(correct_labels),accuracy);
