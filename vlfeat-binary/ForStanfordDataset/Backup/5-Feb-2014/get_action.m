clc
close all
clear all
load action_list
path_load = '/home/ssabidi/Desktop/Stanford40/MatlabAnnotations/';
txt_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/txt/';
img_path = '/home/ssabidi/Desktop/Stanford40/JPEGImages/';

%generate file name
for iter_actions=1:40
    current_label = iter_actions
    load_filename = strcat(path_load,'annotation_',action_list{iter_actions},'.mat');
    load(load_filename);
    for iter_files=1:length(annotation)
        incomplete_img_filename =  annotation{iter_files}.imageName
        [path_to_load,filename_wo_ext,ext] = fileparts(load_filename);
        %generate txt_filename
        %generate image filename
        %dump code from script 2. Now we can have action labels
    end
end
