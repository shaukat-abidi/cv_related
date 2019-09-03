%savng filenames in the order as it was read while generating superpixels.
%We need this to our separate data for testing/training
clc
close all
clear all


path_load = '/home/ssabidi/Desktop/Stanford40/MatlabAnnotations/';
txt_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/txt/';
img_path = '/home/ssabidi/Desktop/Stanford40/JPEGImages/';
fid_train_path = '/home/ssabidi/Desktop/Stanford40/ImageSplits/';

iter_actions = 40;%JUST CHANGE THIS
focused_class = 'c40';%JUST CHANGE THIS

load_filename_1 = strcat(path_load,'action_list.mat');
load(load_filename_1,'-mat');

action_name=action_list{iter_actions};

train_filename=strcat(action_name,'_train.txt');
test_filename=strcat(action_name,'_test.txt');

load_filename_fid_train = strcat(fid_train_path,train_filename);
load_filename_fid_test = strcat(fid_train_path,test_filename);

%loading files
ssplit_train=importdata(load_filename_fid_train);%suggested split for trainning files
ssplit_test=importdata(load_filename_fid_test);%suggested split for test files

filename_in_order = [];%This will contain the files (exactly matching to Qids obtained for generating train
                       % and test sequences)
filename_not_found = [];%Filenames missing due to (1) not being an RGB image (2) Processed unsuccessfully by Felzenschwalb segmentation algorithm

fid=0;

%generate file name
    current_label = iter_actions
    load_filename = strcat(path_load,'annotation_',action_list{iter_actions},'.mat');
    load(load_filename);
    
    for iter_files=1:length(annotation)
        incomplete_img_filename =  annotation{iter_files}.imageName;
        [path_to_load,filename_wo_ext,ext] = fileparts(incomplete_img_filename);
        
        %generate txt_filename
        text_filename = strcat(txt_path,filename_wo_ext,'.txt')
                
        %generate image filename
        img_filename = strcat(img_path,filename_wo_ext,'.jpg');
        
        %preparation for D-Sift
        I = imread(img_filename) ;
        %%%%%to avoid error%%%%%%%
        if (size(I,3) ~=3)
            fprintf(' \n Image doesnt contain rgb channels . Skipping this file \n');
            continue;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%
       
              
        if (exist(text_filename,'file') == 2)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %dump code from script 2. Now we can have action labels
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
              %file_ID: For latent SVM Training file
              fid = fid + 1
              filename_in_order = [filename_in_order;incomplete_img_filename];    
           
        else
            fprintf('text file doesnt exist. \n');
            filename_not_found = [filename_not_found;incomplete_img_filename];

            continue;
        end
        
    end    
    
%USE filename_in_order for further processing

fprintf('calling after_getting_filenames.m . Change filenames and folders in the next script otherwise files would be overwritten. Press Enter to conitnue .. \n');
pause;
after_getting_filenames;
