%%This script will check files if they exist
%%proceed to script 4 for further work

clc
close all
clear all
load action_list
path_load = '/home/ssabidi/Desktop/Stanford40/MatlabAnnotations/';
txt_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/txt/';
img_path = '/home/ssabidi/Desktop/Stanford40/JPEGImages/';

%generate file name
for iter_actions=34:35 %40
    current_label = iter_actions;
    load_filename = strcat(path_load,'annotation_',action_list{iter_actions},'.mat');
    load(load_filename);
    for iter_files=1:length(annotation)
        incomplete_img_filename =  annotation{iter_files}.imageName
        [path_to_load,filename_wo_ext,ext] = fileparts(incomplete_img_filename);
        
        %generate txt_filename
        text_filename = strcat(txt_path,filename_wo_ext,'.txt')
                
        %generate image filename
        img_filename = strcat(img_path,filename_wo_ext,'.jpg')
        
        if (exist(text_filename,'file') == 2)
            %dump code from script 2. Now we can have action labels

        else
            fprintf('text file doesnt exist. \n');
            continue;
        end
    end
end
