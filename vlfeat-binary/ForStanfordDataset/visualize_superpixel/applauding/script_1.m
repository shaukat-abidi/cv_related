clear all
close all
clc

addpath('/home/ssabidi/Shaukat/vlfeat-binary/toolbox/staticAR/functions');

load('applauding_178.mat');
scores = load('superpixel_scores_applauding_178.csv');
scores_sorted = sort(scores,'descend');
assert(length(scores) == length(labels) );
superpixel_id=[];
for i=1:length(scores)
    score_selected=scores_sorted(i);
    [r,c]=find(scores(:) == score_selected);
    superpixel_id = [superpixel_id;r]; %It contains ids for superpixels that are scored highest
end

% Corresponding qid for applauding_1.jpg is qid:44 in train 1vs40.txt(5 qids)
% Corresponding qid for applauding_211.jpg is qid:24 in train 1vs40.txt(5 qids)

% for visualisation only
%generated_image = gen_segmented_image(rgb_img,histogram_vectors,total_no_segments,filename_wo_ext );
%seg_no = superpixel_id(1);
%[pixel_list] = get_seg_pl(segment_list,seg_no,image_pixel_list);

histogram_vectors_refined = repmat(struct('pixels', []), length(labels), 1);
j=1;
for i=1:length(histogram_vectors)
    if (histogram_vectors(i).valid_segment ~= -1)
        histogram_vectors_refined(j).pixels = histogram_vectors(i).pixels;
        j=j+1;
    end
end

pixel_list=[];
limit_superpixels = ceil(0.20*length(labels));
for i=1:limit_superpixels
    seg_no=superpixel_id(i);%Select top 20
    %[temp_pixel_list] = get_seg_pl(segment_list,seg_no,image_pixel_list);
    [temp_pixel_list] =  histogram_vectors_refined(seg_no).pixels;
    pixel_list = [pixel_list;temp_pixel_list];
end
imshow(rgb_img);
hold on
view_segments(rgb_img,pixel_list)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
