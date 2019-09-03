%STEP-1: Accumulate training_fids in sequence (get from ssplit_train)
desired_train_fids = [];
discarded_train_fids = [];

desired_index = -5;
for iter_train_fids = 1:length(ssplit_train)
    desired_index = find_index(filename_in_order,ssplit_train{iter_train_fids});
    if(desired_index ~= -1 )
       desired_train_fids = [desired_train_fids;desired_index]; 
    else
       discarded_train_fids = [discarded_train_fids;iter_train_fids];
    end
    
end

%STEP-2: Accumulate test_fids in sequence (get from ssplit_test)
desired_test_fids = [];
discarded_test_fids = [];

desired_index = -5;
for iter_test_fids = 1:length(ssplit_test)
    desired_index = find_index(filename_in_order,ssplit_test{iter_test_fids});
    if(desired_index ~= -1 )
       desired_test_fids = [desired_test_fids;desired_index]; 
    else
       discarded_test_fids = [discarded_test_fids;iter_test_fids];
    end
    
end

% discarded_train_fids: It contains line number of classname_train.txt 
% (example: applauding_train.txt) which contains the name of discarded file
% Same is the case with discarded_test_fids.
% We need to save 1) desired_train_fids 2) desired_test_fids


save_path = strcat('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/output_by_SVMMultiClass/input_for_LSSVM/qids_according_to_split/',focused_class,'/');
[SUCCESS,MESSAGE,MESSAGEID] = mkdir(save_path)
fName_1 = strcat(save_path,'train_',focused_class,'.txt'); 
fName_2 = strcat(save_path,'test_',focused_class,'.txt'); 
fName_3 = strcat(save_path,'discarded_train_',focused_class,'.txt'); 
fName_4 = strcat(save_path,'discarded_test_',focused_class,'.txt'); 
fName_5 = strcat(save_path,'filenames_inorder_',focused_class,'.txt'); 

%Writing to file
fid = fopen(fName_1,'w');
fprintf('Start writing results to fName_1. \n');
rows = size(desired_train_fids,1); 
for iter_rows=1:rows
    str = strcat(num2str(desired_train_fids(iter_rows)));
    fprintf(fid,'%s\n',str);
end
fclose(fid);   

%Writing to file
fid = fopen(fName_2,'w');
fprintf('Start writing results to fName_2. \n');
rows = size(desired_test_fids,1); 
for iter_rows=1:rows
    str = strcat(num2str(desired_test_fids(iter_rows)));
    fprintf(fid,'%s\n',str);
end
fclose(fid);  

%Writing to file
fid = fopen(fName_3,'w');
fprintf('Start writing results to fName_3. \n');
rows = size(discarded_train_fids,1); 
for iter_rows=1:rows
    str = strcat(num2str(discarded_train_fids(iter_rows)));
    fprintf(fid,'%s\n',str);
end
fclose(fid);  

%Writing to file
fid = fopen(fName_4,'w');
fprintf('Start writing results to fName_4 \n');
rows = size(discarded_test_fids,1); 
for iter_rows=1:rows
    str = strcat(num2str(discarded_test_fids(iter_rows)));
    fprintf(fid,'%s\n',str);
end
fclose(fid);

%Writing to file
fid = fopen(fName_5,'w');
fprintf('Start writing results to fName_5 \n');
rows = size(filename_in_order,1); 
for iter_rows=1:rows
    str = strcat(num2str(filename_in_order(iter_rows,:)));
    fprintf(fid,'%s\n',str);
end
fclose(fid);  


fprintf('done\n');
