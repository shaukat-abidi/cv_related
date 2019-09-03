clc
close all
clear all


path_load = '/home/ssabidi/Shaukat/Stanford40/MatlabAnnotations/';
txt_path = '/home/ssabidi/Shaukat/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/txt/';
img_path = '/home/ssabidi/Shaukat/Stanford40/JPEGImages/';
save_path = '/home/ssabidi/Shaukat/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/';

%Adding path to functions directory
addpath('/home/ssabidi/Shaukat/vlfeat-binary/toolbox/staticAR/functions');


load_filename_1 = strcat(save_path,'kmeans_dictionary/400_clusters_for_descriptors.mat');
load_filename_2 = strcat(path_load,'action_list.mat');

%loading files
load(load_filename_1,'-mat');  
load(load_filename_2,'-mat');
cluster_centres = double(C);
clear C

% examples: bag-of-words for superpixel with 400 dimensions
%labels:    storing action-labels
%fid: current file-ID for superpixel (desired for LSVM training file)
%fid_accumulated: This file-ID will be written to file 
%superpixel_size: Size of that superpixel in pixels
examples = [];
labels=[];
fid_accumulated = [];
fid = 0;
limit = 0;
superpixel_size=[];

%generate file name
for iter_actions=3:3%40
    current_label = iter_actions
    load_filename = strcat(path_load,'annotation_',action_list{iter_actions},'.mat');
    load(load_filename);
    
    for iter_files=8:8%length(annotation)
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
        rgb_img = I;
        gray_img = rgb2gray(I);

        height = size(I,1);
        width = size(I,2);
        I = single(rgb2gray(I)) ;
        
              
        if (exist(text_filename,'file') == 2)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %dump code from script 2. Now we can have action labels
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
              %file_ID: For latent SVM Training file
              fid = fid + 1
            %%%%%%%%%%%%%%Read Text File containing %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%super pixel segmentation information%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
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
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%Read image file and calculate SIFT descriptors%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                      
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Accumulate D-Sift descriptors from current image 
            

            fprintf('Calculating D-Sift for %s ', strcat(filename_wo_ext,'.jpg'));


            binSize = 4;
            magnif = 3 ;
            Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;

            [f1, d1] = vl_dsift(Is, 'size', binSize);

            fprintf('.');

            binSize = 8;
            magnif = 3 ;
            Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;

            [f2, d2] = vl_dsift(Is, 'size', binSize);

            fprintf('.');

            binSize = 16;
            magnif = 3 ;
            Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;

            [f3, d3] = vl_dsift(Is, 'size', binSize);

            features = [f1 f2 f3];
            descriptors = [d1 d2 d3];


            fprintf('. Done \n');
        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%Generating Datastructure for SuperPixel%%%%%%%%%%%%%%%%
            histogram_vectors = repmat(struct('id', -1, 'pixels', [],'object_id', [], 'no_feats', 0, 'closest_cluster' , [] , ...
            'bag_of_words' , zeros(1,400) , 'label' , -1, 'fid', fid, 'valid_segment', -1 ), total_no_segments, 1);
            % histogram_vectors: datastructure for superpixel representation
            % id: superpixel id
            % pixels: feature points constituting superpixel#id
            % object_id: object id for that pixel constituing superpixel#id (it should be same throughout the list
            % otherwise segmentation algorithm is not working fine.)
            % no_feats: total number of feature points in  superpixel#id
            % closest_cluster: ID of closest cluster to which particular pixel belong
            % bag_of_words: normalised BoW representation of superpixel#id
            % fid: file ID (Used for Latent SVM training file)
            % valid_segment: It means that the features of this segment
            % doesn't contain any NaN/InF values

            total_descriptors = size(features,2);

           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            for iter_descr=1:2:total_descriptors
                query = features(:,iter_descr)';
                linear_id = getLinearIndex(query(1),query(2),width,height);
                seg_no = segment_list(linear_id);
                histogram_vectors(seg_no).id = seg_no;
                histogram_vectors(seg_no).pixels = [histogram_vectors(seg_no).pixels;query];
                histogram_vectors(seg_no).no_feats = histogram_vectors(seg_no).no_feats + 1;

                %calculate minimum distance for each feature point
                [distance, cluster_id] = min(vl_alldist(double(descriptors(:,iter_descr)), cluster_centres));
                histogram_vectors(seg_no).closest_cluster = [histogram_vectors(seg_no).closest_cluster;cluster_id];

                %generate histogram vector
                cc_id=cluster_id;%id of closest cluster
                histogram_vectors(seg_no).bag_of_words(cc_id) = histogram_vectors(seg_no).bag_of_words(cc_id) + 1;
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        
        
            for iter_segments = 1:total_no_segments

                if(histogram_vectors(iter_segments).id ~= -1)
                    
                     fprintf('i=%d \n',iter_segments);
                    
                     histogram_vectors(iter_segments).bag_of_words = histogram_vectors(iter_segments).bag_of_words / sum (histogram_vectors(iter_segments).bag_of_words) ;
                     
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
                    
                     %%%%%%filter those superpixels that have NaN or Inf
                     %%%%%%values inside them. this is the case when
                     %%%%%%superpixel consist of one feature only
                     check_inf = 0;
                     check_nan = 0;
                     check_inf = sum(isinf(feature_vector_superpixel));
                     check_nan = sum(isnan(feature_vector_superpixel));
                     
                     if( check_inf == 0 && check_nan == 0 )
                         %%%%%%Accumulate examples,labels,fids,superpixel size%%%%%%%%%%%%
                         histogram_vectors(iter_segments).valid_segment = 1; %This means this segment does'nt c
                         %contain any NaN or Inf 
                         examples = [examples; feature_vector_superpixel];
                         labels = [labels;current_label];
                         fid_accumulated = [fid_accumulated;histogram_vectors(iter_segments).fid];
                         superpixel_size = [superpixel_size;histogram_vectors(iter_segments).no_feats];
                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                     end

                    
                end


            end
        
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        end
        
    end    
    
    % for visualisation only
    %generated_image = gen_segmented_image(rgb_img,histogram_vectors,total_no_segments,filename_wo_ext );
    %seg_no = 1;
    %compare_segments(rgb_img,segment_list,seg_no,image_pixel_list,histogram_vectors);
    %[pixel_list] = get_seg_pl(segment_list,seg_no,image_pixel_list);
    %view_segments(rgb_img,pixel_list)
    %view_segments(rgb_img,histogram_vectors(seg_no).pixels)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
end
% fprintf(' \n Finished processing. \n');
% save_filename = strcat(save_path,'for_training.mat');
% save_filename_1 = strcat(save_path,'fids.mat');
% save_filename_2 = strcat(save_path,'superpixel_size.mat');
% fprintf(' Saving results to file . ');
% 
% save(save_filename,'examples','labels');
% fprintf('.');
% save(save_filename_1,'fid_accumulated');
% fprintf('.');
% save(save_filename_2,'superpixel_size');
% fprintf('. Done \n');