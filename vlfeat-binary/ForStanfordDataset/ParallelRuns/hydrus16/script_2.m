%%Script 6 will store write labels and probs of labels for each superpixels
%%%in L-SVM format which can be used by LIBSVM as well
%%%This script is particularly suitable for Multiclass SVM package

%clc
clear all

tic

%Adding path to functions directory
addpath('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/functions');

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/';
load_filename_1 = strcat(save_path,'for_training_class11_class19.mat');
load_filename_2 = strcat(save_path,'fids_class11_class19.mat');

load(load_filename_1,'-mat');
load(load_filename_2,'-mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dlmwrite('training_file.txt',match_probs,'delimiter',' ');
fName = strcat(save_path,'training_file_class11_class19.txt');

fid = fopen(fName,'w');

fprintf('Start writing results to file. \n');

rows = size(examples,1); 
cols = size(examples,2);
for iter_rows=1:rows
    for iter_cols=1:cols
        if iter_cols == 1
            %str = strcat( num2str( labels(iter_rows) ),32,'qid:',num2str( fid_accumulated(iter_rows) ) );
            str = num2str(1);%writing 1 to every example as its label 
            fprintf(fid,'%s ',str);  
        end
        str = strcat(num2str(iter_cols),':',num2str(examples(iter_rows,iter_cols)));
        fprintf(fid,'%s ',str);
    end
    fprintf(fid,'\n');
    
    if(rem(iter_rows,1000) == 0)
    fprintf('.');
    end
    
end
fclose(fid);   
%dlmwrite('latent_vars_init.txt',match_probs,'delimiter',' ')
writing_file = toc