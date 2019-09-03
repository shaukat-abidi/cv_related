function [ texture_vector ] = texture_features( stats,gray_img )
%Calculate the texture of superpixel
texture_vector = zeros(3,5);
%Filter operation
 %%UL,UR,BL,BR of BBox%%%%%%%
 UL_x = stats.BoundingBox(1);
 UL_y = stats.BoundingBox(2);
 UR_x = stats.BoundingBox(1) + stats.BoundingBox(3);
 UR_y = stats.BoundingBox(2);
 BL_x = UL_x;
 BL_y = UL_y + stats.BoundingBox(4);
 BR_x = UR_x;
 BR_y = BL_y;
 
 %for filter mask
 filter_rows = BL_y - UL_y;
 filter_cols = UR_x - UL_x;
 filter_rows = floor(filter_rows);
 filter_cols = floor(filter_cols);
 
 image_matrix_rows = size(gray_img,1);
 image_matrix_cols = size(gray_img,2);
 
 UL_x = ceil(UL_x); %it will make sure that UL_x is within bounds [1,image_matrix_cols]
 UL_y = ceil(UL_y); %it will make sure that UL_y is within bounds [1,image_matrix_rows]
 UR_x = floor(UR_x);%it will make sure that UR_x is within bounds [1,image_matrix_cols]
 UR_y = ceil(UR_y);%it will make sure that UR_y is within bounds [1,image_matrix_rows]
 BL_x = ceil(BL_x);%it will make sure that BL_x is within bounds [1,image_matrix_cols]
 BL_y = floor(BL_y);%it will make sure that BL_y is within bounds [1,image_matrix_rows]
 BR_x = floor(BR_x);%it will make sure that BR_x is within bounds [1,image_matrix_cols]
 BR_y = floor(BR_y);%it will make sure that BR_y is within bounds [1,image_matrix_rows]
 
 image_data = double(gray_img(UL_y:BL_y,UL_x:UR_x)); %cropped image
 
%  h = fspecial('gaussian', hsize,sigma) returns a rotationally symmetric Gaussian lowpass filter of size hsize with standard deviation sigma (positive). hsize can 
%  be a vector specifying the number of rows and columns in h, or it can be a scalar, in which case h is a square matrix. The default value for hsize is [3 3]; the default value for sigma is 0.5.

% h = fspecial('log', hsize, sigma) returns a rotationally symmetric Laplacian of Gaussian filter of size hsize with standard deviation sigma (positive). hsize can
% be a vector specifying the number of rows and columns in h, or it can be a scalar, in which case h is a square matrix. The default value for hsize is [5
% 5] and 0.5 for sigma.

%h = fspecial('average', hsize) returns an averaging filter h of size hsize. The argument hsize can be a vector specifying the number of rows and columns in h, or it can be a
%scalar, in which case h is a square matrix. The default value for hsize is [3 3].

 
 %filter params
 [s1,s2]=size(image_data);
 
 
 sigma = 0.1;
 h = fspecial('gaussian',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(1,1) = fil_val;
 
 sigma = 0.2;
 h = fspecial('gaussian',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(1,2) = fil_val;
 
 sigma = 0.3;
 h = fspecial('gaussian',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(1,3) = fil_val;
 
 sigma = 0.4;
 h = fspecial('gaussian',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(1,4) = fil_val;
 
 sigma = 0.5;
 h = fspecial('gaussian',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(1,5) = fil_val;
 
 sigma = 0.1;
 h = fspecial('log',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(2,1) = fil_val;
 
 sigma = 0.2;
 h = fspecial('log',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(2,2) = fil_val;
 
 sigma = 0.3;
 h = fspecial('log',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(2,3) = fil_val;
 
 sigma = 0.4;
 h = fspecial('log',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(2,4) = fil_val;
 
 sigma = 0.5;
 h = fspecial('log',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h );
 texture_vector(2,5) = fil_val;
 
 h = fspecial('average', [s1 s2]);
 fil_val = filter_operation( image_data,h );
 texture_vector(3,1) = fil_val;
 texture_vector(3,2) = fil_val;
 texture_vector(3,3) = fil_val;
 texture_vector(3,4) = fil_val;
 texture_vector(3,5) = fil_val;
 
 

end

