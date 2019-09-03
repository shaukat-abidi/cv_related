%%Script 3 will write fids to the file

%clc

tic

%Adding path to functions directory
addpath('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/functions');

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/';
load_filename_1 = strcat(save_path,'fids_class36_class40.mat');

load(load_filename_1,'-mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dlmwrite('training_file.txt',match_probs,'delimiter',' ');
fName = strcat(save_path,'fid_c36_c40.txt');


fid = fopen(fName,'w');

fprintf('Start writing results to file. \n');

rows = size(fid_accumulated,1); 
cols = size(fid_accumulated,2);
for iter_rows=1:rows
    str = strcat(num2str(fid_accumulated(iter_rows)));
    fprintf(fid,'%s\n',str);
    
    if(rem(iter_rows,1000) == 0)
    fprintf('.');
    end    
end
fclose(fid);   
%dlmwrite('latent_vars_init.txt',match_probs,'delimiter',' ')
writing_file = toc
