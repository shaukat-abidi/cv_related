clear all
close all
clc
tic
txt_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/txt/';
img_path = '/home/ssabidi/Desktop/Stanford40/JPEGImages/';

load('learned_weights.mat','-mat');

load('400_clusters_for_descriptors.mat','-mat');
cluster_centres = double(C);
clear C

filePattern = fullfile(txt_path, '*.txt');
listfiles = dir(filePattern);
total_files = length(listfiles);

%%%Training file's data for L-SVM%%%
%data file format: action_label hidden_var_init probability_vals
probability_vals = [];
hidden_var_init = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%For debugging%%%%%%%%%%%%%%%%%%%%%
%%%To check logistic regression classfier
% scores = sigmoid(weights,example)
% scores: 23-valued vector (probability that example belong to 23 classes)
% weights: learned weights by logistic regression classifier
% example: bag-of-words for superpixel with 400 dimensions
examples = [];
scores = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for iter_files=1:1%total_files
    
    %%%%Read Text File containing super pixel segmentation information%%%%%
    text_filename = strcat(txt_path,listfiles(iter_files).name);
    
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
        
        
        %%%%%%%Read image file and calculate SIFT descriptors%%%%%%%%%%%
        [path_to_txt,filename_wo_ext,ext] = fileparts(text_filename);
        img_filename = strcat(img_path,filename_wo_ext,'.jpg');
        
        I = imread(img_filename);
        height = size(I,1);
        width = size(I,2);
        I = single(rgb2gray(I)) ;
        binSize = 8 ;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
        [features, descriptors] = vl_dsift(Is, 'size', binSize) ;
        
        histogram_vectors = repmat(struct('id', -1, 'pixels', [],'object_id', [], 'no_feats', 0, 'closest_cluster' , [] , ...
        'bag_of_words' , zeros(1,400) , 'label' , -1 ), total_no_segments, 1);
        % histogram_vectors: datastructure for superpixel representation
        % id: superpixel id
        % pixels: feature points constituting superpixel#id
        % object_id: object id for that pixel constituing superpixel#id (it should be same throughout the list
        % otherwise segmentation algorithm is not working fine.)
        % no_feats: total number of feature points in  superpixel#id
        % closest_cluster: ID of closest cluster to which particular pixel belong
        % bag_of_words: normalised BoW representation of superpixel#id
        
        total_descriptors = size(features,2);
        debug_matrix = [] ; %[superpixel_id label]
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
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

            mappedD = histogram_vectors(iter_segments).bag_of_words*w;
            probability_vals = [probability_vals;mappedD.data];
            
            labels = mappedD*labeld;
            hidden_var_init = [hidden_var_init;labels];
            
            buffer = [iter_segments , labels];% storing label for each superpixel
            debug_matrix = [debug_matrix;buffer];% storing label for each superpixel
            
            histogram_vectors(iter_segments).label = labels;   

            end


        end

        
end

%%for debugging logistic-regression classifier
weights = []; % weights for 23 classifiers(400x23)
offsets = []; % offsets for 23 classifiers (1 x 23)

for iter_classifier=1:23
    weights = [weights, w.data{iter_classifier}.data.rot];
    offsets = [offsets;w.data{iter_classifier}.data.offset];
end

% sigmoid( theta,offset,x )
% checking example#168
% x=examples(168,:);
% sigmoid(weights(:,1),offsets(1),x') - for class 1
% sigmoid(weights(:,2),offsets(2),x') - for class 2
% to_match=sigmoid(weights,offsets,x);

% logistic classifier for speed
match_probs = [];%calculating probs of each example
                 %manually using logistic function
match_labels = [];%Labels for superpixel
for iter_examples = 1:size(examples,1)
    x=examples(iter_examples,:);
    scores=sigmoid(weights,offsets,x');
    [val,ind] = max(scores);
    match_probs = [match_probs;scores'];
    match_labels = [match_labels;ind];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

total_time = toc