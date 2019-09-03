%Accumulate Dense-Sift for All-Images with 23-Classes
clear all
close all
clc

strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/org/';
text_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/txt/';
gt_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/GT/';
save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForMSRDataset/load_files/';

%%%%initialise cell array for storing sift descriptors %%%%
descriptors_bag = cell(591,1);

i=0;%variable used for indexing descriptors bag

for iter_categories = 1:20
    if(iter_categories == 1)
        lim_img = 30;
    end
    if(iter_categories == 2)
        lim_img = 30;
    end
    if(iter_categories == 3)
        lim_img = 30;
    end
    if(iter_categories == 4)
        lim_img = 30;
    end
    if(iter_categories == 5)
        lim_img = 30;
    end
    if(iter_categories == 6)
        lim_img = 30;
    end
    if(iter_categories == 7)
        lim_img = 30;
    end
    if(iter_categories == 8)
        lim_img = 30;
    end
    if(iter_categories == 9)
        lim_img = 30;
    end
    if(iter_categories == 10)
        lim_img = 32;
    end
    if(iter_categories == 11)
        lim_img = 30;
    end
    if(iter_categories == 12)
        lim_img = 34;
    end
    if(iter_categories == 13)
        lim_img = 30;
    end
    if(iter_categories == 14)
        lim_img = 30;
    end
    if(iter_categories == 15)
        lim_img = 24;
    end
    if(iter_categories == 16)
        lim_img = 30;
    end
    if(iter_categories == 17)
        lim_img = 30;
    end
    if(iter_categories == 18)
        lim_img = 30;
    end
    if(iter_categories == 19)
        lim_img = 30;
    end
    if(iter_categories == 20)
        lim_img = 21;
    end
    
    for iter_img = 1:lim_img
        tic
        filename = strcat(strpath,num2str(iter_categories),'_',num2str(iter_img),'_s.bmp');
        text_filename = strcat(text_strpath,num2str(iter_categories),'_',num2str(iter_img),'_s_out.txt');
        gt_filename = strcat(gt_strpath,num2str(iter_categories),'_',num2str(iter_img),'_s_GT.bmp');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%Accumulating D-Sift%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
         i=i+1; %index for descriptors bag
        
        % Accumulate D-Sift descriptors from all images 
        I = imread(filename) ;
        I = single(rgb2gray(I)) ;
    
        binSize = 4;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
    
        [f1, d1] = vl_dsift(Is, 'size', binSize);
        
        binSize = 8;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
    
        [f2, d2] = vl_dsift(Is, 'size', binSize);
        
        binSize = 16;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
    
        [f3, d3] = vl_dsift(Is, 'size', binSize);
    
        descriptors_bag{i} = [d1 d2 d3]; %accumulating dense-sift points in this bag
       
     
        fprintf('category=%d image# =%d \n',iter_categories,iter_img);
        dsift=toc

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
save_filename_1 = strcat(save_path,'dsift_dataset.mat');
save(save_filename_1,'descriptors_bag','-v7.3');

