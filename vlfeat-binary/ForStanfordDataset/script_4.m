%%Script 4 will write labels to file

tic

%Adding path to functions directory
addpath('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/functions');

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/';
load_filename_1 = strcat(save_path,'for_training_class36_class40.mat');

load(load_filename_1,'-mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dlmwrite('training_file.txt',match_probs,'delimiter',' ');
fName = strcat(save_path,'label_c36_c40.txt');


fid = fopen(fName,'w');

fprintf('Start writing results to file. \n');

rows = size(labels,1); 
cols = size(labels,2);
for iter_rows=1:rows
    str = strcat(num2str(labels(iter_rows)));
    fprintf(fid,'%s\n',str);
    
    if(rem(iter_rows,1000) == 0)
    fprintf('.');
    end
end
fclose(fid);   
%dlmwrite('latent_vars_init.txt',match_probs,'delimiter',' ')
writing_file = toc
