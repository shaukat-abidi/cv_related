%%This script will check files if they exist
%%proceed to script 4 for further work
tic
clc
close all
clear all

load action_list

load('400_clusters_for_descriptors.mat','-mat');
cluster_centres = double(C);
clear C

path_load = '/home/ssabidi/Desktop/Stanford40/MatlabAnnotations/';
txt_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/txt/';
img_path = '/home/ssabidi/Desktop/Stanford40/JPEGImages/';
save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/';

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
for iter_actions=1:5%40
    current_label = iter_actions
    load_filename = strcat(path_load,'annotation_',action_list{iter_actions},'.mat');
    load(load_filename);
    if (iter_actions ==1)
        limit=20;
    else
        limit=5;
    end
        
    for iter_files=1:limit%length(annotation)
        tic
        incomplete_img_filename =  annotation{iter_files}.imageName;
        [path_to_load,filename_wo_ext,ext] = fileparts(incomplete_img_filename);
        
        %generate txt_filename
        text_filename = strcat(txt_path,filename_wo_ext,'.txt')
                
        %generate image filename
        img_filename = strcat(img_path,filename_wo_ext,'.jpg');
        
              
        if (exist(text_filename,'file') == 2)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %dump code from script 2. Now we can have action labels
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
              %file_ID: For latent SVM Training file
              fid = fid + 1
            %%%%%%%%%%%%%%Read Text File containing %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%super pixel segmentation information%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            [ segment_list,image_pixel_list,A,total_no_segments ] = readFile(text_filename);
            % segment_list:     (Nx1) labels each pixel with segment number
            % image_pixel_list: (Nx2) pixel index
            % A               : (Nx3) original parsed file in matrix form
            % total_no_segments: (1x1) total number of segments found in original parsed file 

            % IMPORTANT: Changing the pixels index standard to Matlab's Standard
            % Now the origin is shifted to (1,1) rather than (0,0)
            % This will help us indexing linear addressing
            image_pixel_list(:,1) = image_pixel_list(:,1) + 1;
            image_pixel_list(:,2) = image_pixel_list(:,2) + 1;
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%Read image file and calculate SIFT descriptors%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                      
            I = imread(img_filename);
            height = size(I,1);
            width = size(I,2);
            I = single(rgb2gray(I)) ;
            binSize = 8 ;
            magnif = 3 ;
            Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
            [features, descriptors] = vl_dsift(Is, 'size', binSize) ;

            histogram_vectors = repmat(struct('id', -1, 'pixels', [],'object_id', [], 'no_feats', 0, 'closest_cluster' , [] , ...
            'bag_of_words' , zeros(1,400) , 'label' , -1, 'fid', fid ), total_no_segments, 1);
            % histogram_vectors: datastructure for superpixel representation
            % id: superpixel id
            % pixels: feature points constituting superpixel#id
            % object_id: object id for that pixel constituing superpixel#id (it should be same throughout the list
            % otherwise segmentation algorithm is not working fine.)
            % no_feats: total number of feature points in  superpixel#id
            % closest_cluster: ID of closest cluster to which particular pixel belong
            % bag_of_words: normalised BoW representation of superpixel#id
            % fid: file ID (Used for Latent SVM training file)

            total_descriptors = size(features,2);
           
            
            for iter_descr=1:total_descriptors                
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
                cc_id = cluster_id; %closest cluster id
                histogram_vectors(seg_no).bag_of_words(cc_id) = histogram_vectors(seg_no).bag_of_words(cc_id) + 1;
            end

        
        
        
            for iter_segments = 1:total_no_segments

                if(histogram_vectors(iter_segments).id ~= -1)

                    %%%%%%Normalise bag of words%%%%%%%%%%%%
                     histogram_vectors(iter_segments).bag_of_words = histogram_vectors(iter_segments).bag_of_words / sum (histogram_vectors(iter_segments).bag_of_words) ;
                     examples = [examples; histogram_vectors(iter_segments).bag_of_words];
                     labels = [labels;current_label];
                     fid_accumulated = [fid_accumulated;histogram_vectors(iter_segments).fid];
                     superpixel_size = [superpixel_size;histogram_vectors(iter_segments).no_feats];
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

                    
                end


            end
        
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        
        
        else
            fprintf('text file doesnt exist. \n');
            continue;
        end
        
    one_file_operation=toc    
    end    
   
end
save_filename = strcat(save_path,'for_training.mat');
save_filename_1 = strcat(save_path,'fids.mat');
save_filename_2 = strcat(save_path,'superpixel_size.mat');
save(save_filename,'examples','labels');
save(save_filename_1,'fid_accumulated');
save(save_filename_2,'superpixel_size');


gen_BOW = toc
