%%Script 6 will store write labels and probs of labels for each superpixels
%%%in L-SVM format which can be used by LIBSVM as well

%clc
close all
clear all

tic

%Adding path to functions directory
addpath('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/functions');

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForMSRDataset/load_files/';
load_filename_1 = strcat(save_path,'filtered_trainData.mat');


load(load_filename_1,'-mat');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dlmwrite('training_file.txt',match_probs,'delimiter',' ');
fName = strcat(save_path,'training_file.txt');

fid = fopen(fName,'w');


rows = size(trainingData,1); 
cols = size(trainingData,2);
for iter_rows=1:rows
    for iter_cols=1:cols
        if iter_cols == 1
            fprintf(fid,'%s ', num2str( label(iter_rows) ) );  
        end
        str = strcat(num2str(iter_cols),':',num2str(trainingData(iter_rows,iter_cols)));
        fprintf(fid,'%s ',str);
    end
    fprintf(fid,'\n');
end
fclose(fid);   
%dlmwrite('latent_vars_init.txt',match_probs,'delimiter',' ')
writing_file = toc