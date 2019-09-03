clear all
close all
clc
strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/org/';
text_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/txt/';
gt_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/GT/';
i=1;

load('400_clusters_for_descriptors.mat');
cluster_centres = double(C);
clear C

%Preparing Training Set
label = [];
trainingData = [];

for iter_categories = 1:20 %goes from 1:20
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
        
        [ segment_list,image_pixel_list,A,total_no_segments ] = readFile(text_filename);
        % segment_list:     (Nx1) labels each pixel with segment number
        % image_pixel_list: (Nx2) pixel index
        % A               : (Nx3) original parsed file in matrix
        % total_no_segments: (1x1) total number of segments found in original parsed file 

        % IMPORTANT: Changing the pixels index standard to Matlab's Standard
        % Now the origin is shifted to (1,1) rather than (0,0)
        % This will help us indexing linear addressing
        image_pixel_list(:,1) = image_pixel_list(:,1) + 1;
        image_pixel_list(:,2) = image_pixel_list(:,2) + 1;
        
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%Accumulating D-Sift%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        % Accumulate D-Sift descriptors from all images 
        I = imread(filename) ;
        height = size(I,1);
        width = size(I,2);
        groundtruth_image=imread(gt_filename);
        I = single(rgb2gray(I)) ;
    
        binSize = 8 ;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
    
        [features, descriptors] = vl_dsift(Is, 'size', binSize);
    
           
        fprintf('iter_cat = %d , image = %d ... \n',iter_categories,iter_img);

        dsift=toc

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%Generating Datastructure for SuperPixel%%%%%%%%%%%%%%%%
        histogram_vectors = repmat(struct('id', -1, 'pixels', [],'object_id', [], 'no_feats', 0, 'closest_cluster' , [] , ...
        'bag_of_words' , zeros(1,400) ), total_no_segments, 1);
        % histogram_vectors: datastructure for superpixel representation
        % id: superpixel id
        % pixels: feature points constituting superpixel#id
        % object_id: object id for that pixel constituing superpixel#id (it should be same throughout the list
        % otherwise segmentation algorithm is not working fine.)
        % no_feats: total number of feature points in  superpixel#id
        % closest_cluster: ID of closest cluster to which particular pixel belong
        % bag_of_words: normalised BoW representation of superpixel#id

        total_descriptors = size(features,2);
        
        
        for iter_descr=1:total_descriptors
            query = features(:,iter_descr)';
            linear_id = getLinearIndex(query(1),query(2),width,height);
            seg_no = segment_list(linear_id);
            histogram_vectors(seg_no).id = seg_no;
            histogram_vectors(seg_no).pixels = [histogram_vectors(seg_no).pixels;query];
            histogram_vectors(seg_no).no_feats = histogram_vectors(seg_no).no_feats + 1;

            %Assigning ground truth to every pixel in superpixel
            [red,green,blue] = getPixelValue(groundtruth_image,query);
            obj_id = getGroundTruth(red,green,blue);
            histogram_vectors(seg_no).object_id = [histogram_vectors(seg_no).object_id;obj_id];
            
            %calculate minimum distance for each feature point
            [distance, cluster_id] = min(vl_alldist(double(descriptors(:,iter_descr)), cluster_centres));
            histogram_vectors(seg_no).closest_cluster = [histogram_vectors(seg_no).closest_cluster;cluster_id];

            %generate histogram vector
            cc_id=cluster_id;%id of closest cluster
            histogram_vectors(seg_no).bag_of_words(cc_id) = histogram_vectors(seg_no).bag_of_words(cc_id) + 1;
        end
        

        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%Assign Label to each superpixel of image%%%%%%%%%%%
        
        for iter_segments = 1:total_no_segments
            
            %%%%%%Normalise bag of words%%%%%%%%%%%%
             histogram_vectors(iter_segments).bag_of_words = histogram_vectors(iter_segments).bag_of_words / sum (histogram_vectors(iter_segments).bag_of_words) ;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
                        
             if(histogram_vectors(iter_segments).no_feats ~= 0)
                 
                [mode_obj, freq_obj] = mode(histogram_vectors(iter_segments).object_id);
                
                if(mode_obj ~= -1)
                    %fprintf('i=%d \n',iter_segments);
                    trainingData = [trainingData;histogram_vectors(iter_segments).bag_of_words];
                    label = [label;mode_obj];
                end
            
             end
            
            
            
            %%%%%Just for visualisation%%%%%
            %valid_pixel_bag=[valid_pixel_bag;histogram_vectors(iter_segments).pixels];
            %view_segments(filename,histogram_vectors(iter_segments).pixels)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
    end
end

save('train_data.mat','trainingData');
save('labels.mat','label');


