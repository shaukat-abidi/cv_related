clear all,close all,clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%First Part of Script%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  strpath = 'C:\Shaukat\vlfeat-binary\toolbox\staticAR\Images\';
%  filenames = dir(strpath); %valid filenames from 3:32
%  i=1;

% tic
% for iter=3:32
%     % Accumulate D-Sift descriptors from all images 
%     img=strcat(strpath,filenames(iter).name);
%     I = imread(img) ;
%     I = single(rgb2gray(I)) ;
% 
%     binSize = 8 ;
%     magnif = 3 ;
%     Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
% 
%     [f, d] = vl_dsift(Is, 'size', binSize) ;
% 
%     descriptors_bag{i} = d; %accumulating dense-sift points in this bag
%     i=i+1;
% end
% d_sift_cal=toc
% %save('bag_for_cars.mat','descriptors_bag');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%Second Part of Script%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply K-Means to cluster above accumulated descriptors
% load('bag_for_cars.mat');
% K=12;
% data=cat(2,descriptors_bag{:});
% tic
% [C] = vl_ikmeans(data,K,'method', 'elkan');
% elkan=toc
% save('12-cluster_for_cars.mat','C');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%Third Part of Script(Nearest Neighbour)%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('bag_for_cars.mat');
% load('12-cluster_for_cars.mat');
% data=cat(2,descriptors_bag{:});
% c=double(C);
% data=double(data);
% tic
% [distance, cluster_id] = min(vl_alldist(data(:,56565), c))  
% nearest_neighbour=toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%Training set for logistic regression%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To read file
[ segment_list,image_pixel_list,A,total_no_segments ] = readFile('7_29_s_out.txt');
% segment_list:     (Nx1) labels each pixel with segment number
% image_pixel_list: (Nx2) pixel index
% A               : (Nx3) original parsed file in matrix
% total_no_segments: (1x1) total number of segments found in original parsed file 

% IMPORTANT: Changing the pixels index standard to Matlab's Standard
% Now the origin is shifted to (1,1) rather than (0,0)
% This will help us indexing linear addressing
image_pixel_list(:,1) = image_pixel_list(:,1) + 1;
image_pixel_list(:,2) = image_pixel_list(:,2) + 1;



car_gt = [64,0,128]; %Ground-Truth value for segment car


%%%%%Get segment's pixel list%%%%%%%%%%%%%
% pixel_list = get_seg_pl(segment_list,77,image_pixel_list) %get pixel list for segment#77
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% To get the segment of pixel "query"%%%%%%%%%%%%%
% query = [113,146];
% seg_no = get_segId(segment_list,image_pixel_list,query);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate histogram for image 
% Step-1: Calculate D-Sift of image 
% Step-2: Get the descriptors location index
% Step-3: Find to which segment does it belong 
% Step-4: Generate histogram vector (having K values. Here K=12)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Step:1
I = imread('input.bmp') ;
groundtruth_image=imread('output.bmp');
%seg_o=imread('segmented_output.ppm');
i=I;
I = single(rgb2gray(I)) ;
binSize = 8 ;
magnif = 3 ;
Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
[features, descriptors] = vl_dsift(Is, 'size', binSize) ;



histogram_vectors = repmat(struct('id', -1, 'pixels', [], 'no_feats', 0, 'closest_cluster' , [] , ...
    'isValid' , 0 , 'isCar' , [], 'bag_of_words' , zeros(1,12) ), total_no_segments, 1);
% histogram_vectors: datastructure for superpixel representation
% id: superpixel id
% pixels: feature points constituting superpixel#id
% no_feats: total number of feature points in  superpixel#id
% closest_cluster: ID of closest cluster to which particular pixel belong
% isValid: superpixel having feature points of cars only (0/1)
% isCar: Identify whether every pixel in this superpixel belong to car or not
% bag_of_words: normalised BoW representation of superpixel#id

total_descriptors = size(features,2);

load('12-cluster_for_cars.mat');
c=double(C);
cluster_centres = c;
clear c; 
clear C;
% data=double(data);
% tic
% [distance, cluster_id] = min(vl_alldist(data(:,56565), c))

for iter_descr=1:total_descriptors
    query = features(:,iter_descr)';
    linear_id = getLinearIndex(query(1),query(2));
    seg_no = segment_list(linear_id);
    histogram_vectors(seg_no).id = seg_no;
    histogram_vectors(seg_no).pixels = [histogram_vectors(seg_no).pixels;query];
    histogram_vectors(seg_no).no_feats = histogram_vectors(seg_no).no_feats + 1;
    
    
    [red green blue] = getPixelValue(groundtruth_image,query);
    if_gt = ifCar(red,green,blue,car_gt);
    histogram_vectors(seg_no).isCar = [histogram_vectors(seg_no).isCar;if_gt];
    
    %calculate minimum distance for each feature point
    [distance, cluster_id] = min(vl_alldist(double(descriptors(:,iter_descr)), cluster_centres));
    histogram_vectors(seg_no).closest_cluster = [histogram_vectors(seg_no).closest_cluster;cluster_id];
    
    %generate histogram vector
    cc_id = histogram_vectors(seg_no).closest_cluster; %closest cluster id
    histogram_vectors(seg_no).bag_of_words(cc_id) = histogram_vectors(seg_no).bag_of_words(cc_id) + 1;
    
end

%filter valid segments and accumulate them in a bag
valid_pixel_bag=[];
bow_with_car = [];
bow_with_nocar = [];

%Preparing Training Set
label = [];
trainingData = [];

for iter_segments = 1:total_no_segments
    %%%%%%Normalise bag of words%%%%%%%%%%%%
     histogram_vectors(iter_segments).bag_of_words = histogram_vectors(iter_segments).bag_of_words / sum (histogram_vectors(iter_segments).bag_of_words) ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%Start filtering superpixels with car only
    size_vector = length(histogram_vectors(iter_segments).isCar); 
    sum_vector = sum(histogram_vectors(iter_segments).isCar(:));
    
    if (sum_vector == size_vector && histogram_vectors(iter_segments).id ~= -1 )
        % histogram_vectors(iter_segments).isValid means it is 
        % a valid superpixel which belongs to region "car" only
        histogram_vectors(iter_segments).isValid = 1;
        bow_with_car = [bow_with_car;histogram_vectors(iter_segments).bag_of_words];
        tainingData = [trainingData;bow_with_car];
        label = [label;1];
        
        %%%%%Just for visualisation%%%%%
        %valid_pixel_bag=[valid_pixel_bag;histogram_vectors(iter_segments).pixels];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    elseif (histogram_vectors(iter_segments).id ~= -1)        
        %Collect BoW with No-Car
        bow_with_nocar = [bow_with_nocar;histogram_vectors(iter_segments).bag_of_words];  
        tainingData = [trainingData;bow_with_nocar];
        label = [label;2];
    end
end


%%%Just to visualise how many feature points does each segment contain%%%%%%
%a=cat(1,histogram_vectors.id,histogram_vectors.no_feats);
%b=reshape(a,total_no_segments,2);
%view_segments(i,histogram_vectors(seg_no).pixels);
%visualise_a=cat(1,histogram_vectors.id,histogram_vectors.no_feats,histogram_vectors.isValid);
%visualise_b=reshape(visualise_a,total_no_segments,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%This script ends her%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now moving to train_classifier.m 
% In this script we have played a lot with one image only. 
% It verifies few functions with visualisation
% We have formulated our data-structure , formed superpixel, formed bag of 
% words and normalised them. Finally, we have gotten a normalised histogram
% vector for superpixel. We have also gotten rid of segments/superpixels with no 
% feature points inside them. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



