%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.
clear all,close all,clc;
load('train_data.mat');
load('labels.mat');

load('12-cluster_for_cars.mat');
c=double(C);
cluster_centres = c;
clear c; 
clear C;

strpath = 'C:\Shaukat\vlfeat-binary\toolbox\staticAR\Images\';
text_strpath = 'C:\Shaukat\vlfeat-binary\toolbox\staticAR\Images\txt\';
gt_strpath = 'C:\Shaukat\vlfeat-binary\toolbox\staticAR\Images\GT\';


new_dataset = dataset(trainingData,label);
desired_segments = [];%segments classified as car

%Learning logistic regression classifier
w=loglc(new_dataset);
% mappedD = new_dataset*w;
% labels = mappedD*labeld;
% figure(1); 
% clf;
% scatterd(new_dataset); 
% plotc(w,'k');

%%%%classify image portions%%%%

for iter_file = 26:26
filename = strcat(strpath,'7_',num2str(iter_file),'_s.bmp');
text_filename = strcat(text_strpath,'7_',num2str(iter_file),'_s_out.txt');
gt_filename = strcat(gt_strpath,'7_',num2str(iter_file),'_s_GT.bmp');

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


I = imread(filename) ;
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

for iter_descr=1:total_descriptors
    query = features(:,iter_descr)';
    linear_id = getLinearIndex(query(1),query(2));
    seg_no = segment_list(linear_id);
    histogram_vectors(seg_no).id = seg_no;
    histogram_vectors(seg_no).pixels = [histogram_vectors(seg_no).pixels;query];
    histogram_vectors(seg_no).no_feats = histogram_vectors(seg_no).no_feats + 1;
    
    
    %calculate minimum distance for each feature point
    [distance, cluster_id] = min(vl_alldist(double(descriptors(:,iter_descr)), cluster_centres));
    histogram_vectors(seg_no).closest_cluster = [histogram_vectors(seg_no).closest_cluster;cluster_id];
    
    %generate histogram vector
    cc_id = histogram_vectors(seg_no).closest_cluster; %closest cluster id
    histogram_vectors(seg_no).bag_of_words(cc_id) = histogram_vectors(seg_no).bag_of_words(cc_id) + 1;
    
end


for iter_segments = 1:total_no_segments
    if(histogram_vectors(iter_segments).id ~= -1)
    %%%%%%Normalise bag of words%%%%%%%%%%%%
     histogram_vectors(iter_segments).bag_of_words = histogram_vectors(iter_segments).bag_of_words / sum (histogram_vectors(iter_segments).bag_of_words) ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    mappedD = histogram_vectors(iter_segments).bag_of_words*w;
    labels = mappedD*labeld;
    
        if (labels == 1)
            desired_segments=[desired_segments;iter_segments];
        end
        
    else
        continue;
    
    end
    
    
end


end

pixels = [];

for iter_seg=1:length(desired_segments)
    index_segment = desired_segments(iter_seg);
    pixels=[pixels;histogram_vectors(index_segment).pixels];
end

img = imread(filename);
view_segments(img,pixels);


