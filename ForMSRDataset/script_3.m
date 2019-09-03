clear all
close all
clc
strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/org/';
text_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/txt/';
gt_strpath = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/Images/GT/';

%Adding path to functions directory
addpath('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/functions');

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForMSRDataset/load_files/';
load_filename_1 = strcat(save_path,'400_clusters_for_descriptors.mat');

i=1;

%loading files
load(load_filename_1,'-mat');  
cluster_centres = double(C);
clear C

%Preparing Training Set (90%)
label = [];
trainingData = [];

%Prepare test Set (10%)
test_label=[];
testData=[];

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
        filename = strcat(strpath,num2str(iter_categories),'_',num2str(iter_img),'_s.bmp');
        debug_filename = strcat(num2str(iter_categories),'_',num2str(iter_img),'_s.bmp');
        text_filename = strcat(text_strpath,num2str(iter_categories),'_',num2str(iter_img),'_s_out.txt');
        gt_filename = strcat(gt_strpath,num2str(iter_categories),'_',num2str(iter_img),'_s_GT.bmp');
        
        [ segment_list,image_pixel_list,A,total_no_segments,segment_density ] = readFile(text_filename);
        % segment_list:     (Nx1) labels each pixel with segment number
        % image_pixel_list: (Nx2) pixel index
        % A               : (Nx3) original parsed file in matrix
        % total_no_segments: (1x1) total number of segments found in original parsed file 
        % segment_density:   (total_no_segments x 2) Column-1 contains segment# and
		% Column-2 contains total pixels inside corresponding segment 
		
        % IMPORTANT: Changing the pixels index standard to Matlab's Standard
        % Now the origin is shifted to (1,1) rather than (0,0)
        % This will help us indexing linear addressing
        image_pixel_list(:,1) = image_pixel_list(:,1) + 1;
        image_pixel_list(:,2) = image_pixel_list(:,2) + 1;
        
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%Accumulating D-Sift%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        % Accumulate D-Sift descriptors of current image 
        I = imread(filename) ;
        rgb_img = I;
        gray_img = rgb2gray(I);

        height = size(I,1);
        width = size(I,2);
        groundtruth_image=imread(gt_filename);
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
        
        features = [f1 f2 f3];
        descriptors = [d1 d2 d3];
    
          
        fprintf('iter_cat = %d , image = %d ... \n',iter_categories,iter_img);
         if(iter_img < (lim_img - 3) )
            fprintf('for training data ... \n');      
         else
            fprintf('for test data ... \n');      
         end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%Generating Datastructure for SuperPixel%%%%%%%%%%%%%%%%
        histogram_vectors = repmat(struct('id', -1, 'pixels', [],'object_id', [], 'no_feats', 0, 'closest_cluster' , [] , ...
        'bag_of_words' , zeros(1,400), 'valid_object_id', -1 ), total_no_segments, 1);
        % histogram_vectors: datastructure for superpixel representation
        % id: superpixel id
        % pixels: feature points constituting superpixel#id
        % object_id: object id for that pixel constituing superpixel#id (it should be same throughout the list
        % otherwise segmentation algorithm is not working fine.)
        % no_feats: total number of feature points in  superpixel#id
        % closest_cluster: ID of closest cluster to which particular pixel belong
        % bag_of_words: normalised BoW representation of superpixel#id
        % valid_object_id: if not -1, this means this superpixel belongs to
        % valid object class. For debugging purpose only
        
        
        total_descriptors = size(features,2);
        
    
        for iter_descr=1:2:total_descriptors
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
            %Note: It should execute when no.of feats
            %in bag_of_words > 0 . This is corrected 
            %in Stanford Dataset collection
             histogram_vectors(iter_segments).bag_of_words = histogram_vectors(iter_segments).bag_of_words / sum (histogram_vectors(iter_segments).bag_of_words) ;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
                        
             if(histogram_vectors(iter_segments).no_feats ~= 0)
                 
                [mode_obj, freq_obj] = mode(histogram_vectors(iter_segments).object_id);
                
                if(mode_obj ~= -1)
                    %fprintf('i=%d \n',iter_segments);
                    
                    %%%%%%%%Storing label of object as this segment has valid label assignment 
                     histogram_vectors(iter_segments).valid_object_id = mode_obj;
                     
                     %%%%%%%add features here%%%%%                    
                     %Get binary image with those pixels equal to 1 that are
					 %stored in pixel_list 
					 output_img = gen_binary_image(rgb_img,histogram_vectors(iter_segments).pixels );
					 
					 %figure,imshow(output_img);
					 
					 %Get the stats for superpixel in output image
					 stats=superpixel_props(output_img);
					 fvec = generate_features(rgb_img,histogram_vectors(iter_segments).pixels,stats);
					 texture_vector = texture_features(stats,gray_img);
					 appearance_vector = [fvec(1,:) fvec(2,:) fvec(3,:) fvec(4,:) texture_vector(1,:) texture_vector(2,:) texture_vector(3,:)];
				     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				     
				     feature_vector_superpixel = [histogram_vectors(iter_segments).bag_of_words appearance_vector];
				     
                     if(iter_img < (lim_img - 3) )
                        trainingData = [trainingData;feature_vector_superpixel];
                        label = [label;mode_obj];        
                     else
                         testData = [testData;feature_vector_superpixel];
                         test_label = [test_label;mode_obj];                         
                     end
                     
                end
            
             end
            
            
            
            %%%%%Just for visualisation%%%%%
            %valid_pixel_bag=[valid_pixel_bag;histogram_vectors(iter_segments).pixels];
            %view_segments(filename,histogram_vectors(iter_segments).pixels)
            if (iter_img == 1 || iter_img == 9 || iter_img == 23)
                view_labelled_image(rgb_img,histogram_vectors,total_no_segments,debug_filename )
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
    end
end

save_filename_1 = strcat(save_path,'train_data.mat');
save_filename_2 = strcat(save_path,'labels.mat');
save_filename_3 = strcat(save_path,'test_data.mat');
save_filename_4 = strcat(save_path,'test_labels.mat');
save(save_filename_1,'trainingData','-v7.3');
save(save_filename_2,'label','-v7.3');
save(save_filename_3,'testData','-v7.3');
save(save_filename_4,'test_label','-v7.3');
