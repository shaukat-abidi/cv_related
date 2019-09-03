clear all
close all
clc

addpath('/home/ssabidi/Shaukat/vlfeat-binary/toolbox/staticAR/functions');

load('brushing_teeth_008.mat');
scores = load('scores_brushing_008.txt');
scores_sorted = sort(scores,'descend');
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
for i=1:20
    seg_no=superpixel_id(i);%Select top 20
    %[temp_pixel_list] = get_seg_pl(segment_list,seg_no,image_pixel_list);
    [temp_pixel_list] =  histogram_vectors_refined(seg_no).pixels;
    pixel_list = [pixel_list;temp_pixel_list];
end
view_segments(rgb_img,pixel_list)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
