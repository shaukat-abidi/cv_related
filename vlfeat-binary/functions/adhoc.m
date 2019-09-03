close all,clear all,clc

addpath('/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/functions');
img=imread('18_27_s.bmp');
% To read file
[ segment_list,image_pixel_list,A,total_no_segments,segment_density ] = readFile('18_27_s_out.txt');
% segment_list:      (Nx1) labels each pixel with segment number
% image_pixel_list:  (Nx2) pixel index
% A               :  (Nx3) original parsed file in matrix form
% total_no_segments: (1x1) total number of segments found in original parsed file 
% segment_density:   (total_no_segments x 2) Column-1 contains segment# and
% Column-2 contains total pixels inside corresponding segment 

% IMPORTANT: Changing the pixels index standard to Matlab's Standard
% Now the origin is shifted to (1,1) rather than (0,0)
% This will help us indexing linear addressing
image_pixel_list(:,1) = image_pixel_list(:,1) + 1;
image_pixel_list(:,2) = image_pixel_list(:,2) + 1;

%%%%%Get segment's pixel list%%%%%%%%%%%%%
 pixel_list = get_seg_pl(segment_list,97,image_pixel_list); %get pixel list for segment#77
 
 %viewing the segment
 %view_segments(img,pixel_list); 
 
 %Get binary image with those pixels equal to 1 that are
 %stored in pixel_list 
 output_img = gen_binary_image( img,pixel_list );
 
 %figure,imshow(output_img);
 
 %Get the stats for superpixel in output image
 stats=superpixel_props(output_img);
 
 %verifying binary image using pixelList from stats 
 %verify_output = gen_binary_image(img,stats.PixelList);
 
 
 %figure,imshow(verify_output);
 
 %draw objects(desired features) obtained from 
 %stas over colored image
%  draw_object(img,stats);
 
 %%UL,UR,BL,BR of BBox%%%%%%%
%  UL_x = stats.BoundingBox(1);
%  UL_y = stats.BoundingBox(2);
%  UR_x = stats.BoundingBox(1) + stats.BoundingBox(3);
%  UR_y = stats.BoundingBox(2);
%  BL_x = UL_x;
%  BL_y = UL_y + stats.BoundingBox(4);
%  BR_x = UR_x;
%  BR_y = BL_y;
%  
%  %for filter mask
%  filter_rows = BL_y - UL_y;
%  filter_cols = UR_x - UL_x;
%  filter_rows = floor(filter_rows);
%  filter_cols = floor(filter_cols);
%  
%  gray_img = rgb2gray(img);
%  image_matrix_rows = size(gray_img,1);
%  image_matrix_cols = size(gray_img,2);
%  
%  UL_x = ceil(UL_x); %it will make sure that UL_x is within bounds [1,image_matrix_cols]
%  UL_y = ceil(UL_y); %it will make sure that UL_y is within bounds [1,image_matrix_rows]
%  UR_x = floor(UR_x);%it will make sure that UR_x is within bounds [1,image_matrix_cols]
%  UR_y = ceil(UR_y);%it will make sure that UR_y is within bounds [1,image_matrix_rows]
%  BL_x = ceil(BL_x);%it will make sure that BL_x is within bounds [1,image_matrix_cols]
%  BL_y = floor(BL_y);%it will make sure that BL_y is within bounds [1,image_matrix_rows]
%  BR_x = floor(BR_x);%it will make sure that BR_x is within bounds [1,image_matrix_cols]
%  BR_y = floor(BR_y);%it will make sure that BR_y is within bounds [1,image_matrix_rows]
%  
%  image_data = double(gray_img(UL_y:BL_y,UL_x:UR_x)); %cropped image
%  
%  %filter params
%  [s1,s2]=size(image_data);
%  sigma = 0.5;
%  h = fspecial('gaussian',[s1 s2],sigma);
%  fil_val = filter_operation( image_data,h )
 
 %hold on
 %scatter(BR_x,BR_y,'filled');
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %%%%%Feature-Generation%%%%%
 gray_img = rgb2gray(img);
 tic
 fvec = generate_features(img,pixel_list,stats);
 texture_vector = texture_features(stats,gray_img);
 gen_features = toc
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%%%Gaussian filter implementation%%%%%%%%%
% sigma = 0.5
% denom = sigma * sqrt(2*pi);
% 
% sum_h =  exp(-1 *  (0) / (2 * sigma.^2) ) ...
%             + exp(-1 *  ( 1 ) / (2 * sigma.^2) ) ...
%             + exp(-1 * (4) / (2 * sigma.^2) ) ...
%             + exp(-1 *  ( 1 ) / (2 * sigma.^2) ) ...
%             + exp(-1 *  ( 2 ) / (2 * sigma.^2) ) ...
%             + exp(-1 *  ( 5 ) / (2 * sigma.^2) ) ...
%             + exp(-1 *  ( 4 ) / (2 * sigma.^2) ) ...
%             + exp(-1 *  ( 5 ) / (2 * sigma.^2) ) ...
%             + exp(-1 *  ( 8 ) / (2 * sigma.^2) );
%         
% vec_h =  [ exp(-1 *  (2)   / (2 * sigma.^2) ) , exp(-1 * (1)  / (2 * sigma.^2) ) , exp(-1 *  (2)  / (2 * sigma.^2) ) 
%            exp(-1 *  (1)  /  (2 * sigma.^2) ) , exp(-1 * (0)  / (2 * sigma.^2) ) , exp(-1 *  (1) / (2 * sigma.^2) ) 
%            exp(-1 *  (2) /  (2 * sigma.^2) ) , exp(-1 *  (1) / (2 * sigma.^2) ) , exp(-1 *  (2) / (2 * sigma.^2) )]
% sum_vech = sum(vec_h(:))
% h = fspecial('gaussian',[3 3],sigma)
% vec_h = vec_h / sum_vech 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 