%%Script 6 will store write labels and probs of labels for each superpixels
%%%in L-SVM format

%clc
close all
clear all

tic

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/';
load_filename_1 = strcat(save_path,'for_writingToFile.mat');
load_filename_2 = strcat(save_path,'latent_variable_init.mat');
load_filename_3 = strcat(save_path,'fids.mat');
load_filename_4 = strcat(save_path,'superpixel_size.mat');

load(load_filename_1,'-mat');
load(load_filename_2,'-mat');
load(load_filename_3,'-mat');
load(load_filename_4,'-mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dlmwrite('training_file.txt',match_probs,'delimiter',' ');
fName = strcat(save_path,'training_file.txt');
fName_latentVars = strcat(save_path,'latent_vars_init.txt');
fName_superpixelSize = strcat(save_path,'superpixel_size.txt');

fid = fopen(fName,'w');
fid_latent = fopen(fName_latentVars,'w');
fid_spsize = fopen(fName_superpixelSize,'w');


rows = size(match_probs,1); 
cols = size(match_probs,2);
for iter_rows=1:rows
    for iter_cols=1:cols
        if iter_cols == 1
            str = strcat(num2str(labels(iter_rows)),32,'qid:',num2str(fid_accumulated(iter_rows)));
            fprintf(fid,'%s ',str);  
        end
        str = strcat(num2str(iter_cols),':',num2str(match_probs(iter_rows,iter_cols)));
        fprintf(fid,'%s ',str);
    end
    fprintf(fid,'\n');
    fprintf(fid_latent,'%s\n',num2str(match_latentlabels(iter_rows))); 
    fprintf(fid_spsize,'%s\n',num2str(superpixel_size(iter_rows))); 
end
fclose(fid);   
%dlmwrite('latent_vars_init.txt',match_probs,'delimiter',' ')
writing_file = toc