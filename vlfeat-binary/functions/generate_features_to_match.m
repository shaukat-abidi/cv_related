%function [ feature_vector ] = generate_features( rgb_img,pixel_list,stats )
%pixel_list: contains pixels of superpixel under consideration. 
%pixel stored in pixel list have image convention. Its 1st
%column have values along "vertical axis(where rows change)" and 2nd column 
%have values along "horizontal axis(where columns change)"
% "pixel_list" format:
% |index rows in image data matrix|index columns in image data matrix|
% |                               |                                  |
% |                               |                                  |
% Remember, for scatter function, first argument is variation along
% horizontal axis and second argument is variation along vertical axis

% Why should we consider index rows and index columns? because we want to access data-matrix for
% image. To access data-matrix of image, the first argument is row# and 2nd
% argument is column#, and if we inspect pixel values of an image using imshow(img)'s
% data-cursor, we will see that we can't access those pixel values directly from
% data matrix. This is because, in Image, X:320 Y:1 means col#:320 row#:1
% and if we want to access its value, we need to access like this :
% Image_data_matrix(row#,col#) or Image_data_matrix(Y,X) or Image_data_matrix(1,320)

gray_img = rgb2gray(rgb_img);
ycbcr_img = rgb2ycbcr(rgb_img);
[lab_l,lab_a,lab_b] = RGB2Lab(rgb_img);

lab_l = double(lab_l);
lab_a = double(lab_a);
lab_b = double(lab_b);
gray_img = double(gray_img);

%Generate appearance feature vector

%RGB-vals
r_val = double( rgb_img(:,:,1) );
g_val = double( rgb_img(:,:,2) );
b_val = double(rgb_img(:,:,3) );

%YCbCr-Vals
y_val = double(ycbcr_img(:,:,1));
cb_val = double(ycbcr_img(:,:,2));
cr_val = double(ycbcr_img(:,:,3));


scalar_totpixels = size(pixel_list,1); %total number of pixels in superpixel


sum_r = 0; %sum of red values 
sum_g = 0; %sum of green values
sum_b = 0; %sum of blue values
mean_r = 0; %mean or red color in superpixel
mean_g = 0; %mean or green color in superpixel
mean_b = 0; %mean or blue color in superpixel

sum_lab_l = 0;%sum of l values 
sum_lab_a = 0;%sum of a values
sum_lab_b = 0;%sum of b values
mean_lab_l = 0; %mean of l in superpixel
mean_lab_a = 0; %mean of a in superpixel
mean_lab_b = 0; %mean of b in superpixel

sum_ycbcr_y = 0;%sum of y values 
sum_ycbcr_cb = 0;%sum of cb values
sum_ycbcr_cr = 0;%sum of cr values
mean_ycbcr_y = 0; %mean of y in superpixel
mean_ycbcr_cb = 0; %mean of cb in superpixel
mean_ycbcr_cr = 0; %mean of cr in superpixel

sum_gray = 0;%sum of all gray pixels
mean_gray = 0;%mean of grayed superpixel

%Standard-Deviation
std_rgb_r = 0;
std_rgb_g = 0;
std_rgb_b = 0;
std_lab_l = 0;
std_lab_a = 0;
std_lab_b = 0;
std_ycrcb_y = 0;
std_ycrcb_cr = 0;
std_ycrcb_cb = 0;
std_gr = 0;

%Skewness
sk_rgb_r = 0;
sk_rgb_g = 0;
sk_rgb_b = 0;
sk_lab_l = 0;
sk_lab_a = 0;
sk_lab_b = 0;
sk_ycrcb_y = 0;
sk_ycrcb_cr = 0;
sk_ycrcb_cb = 0;
sk_gr = 0;

%kurtosis
kur_rgb_r = 0;
kur_rgb_g = 0;
kur_rgb_b = 0;
kur_lab_l = 0;
kur_lab_a = 0;
kur_lab_b = 0;
kur_ycrcb_y = 0;
kur_ycrcb_cr = 0;
kur_ycrcb_cb = 0;
kur_gr = 0;

%Shape-Feature
sf_a = 0;
sf_b = 0;
sf_c = 0;

%Location Features
lf_a = 0;
lf_b = 0;
lf_c = 0;
image_centre = zeros(1,2);
image_centre(1) = size(rgb_img,1) / 2;
image_centre(2) = size(rgb_img,2) / 2; 

%generate means
for i = 1:scalar_totpixels
    r = pixel_list(i,1);%row for current pixel in image matrix
    c = pixel_list(i,2);%col for current pixel in image matrix
     
    %fprintf('r: %d\n',r_val(r,c));
    %fprintf('g: %d\n',g_val(r,c));
    %fprintf('b: %d\n',b_val(r,c));
    
    %RGB Pixels
    sum_r = sum_r + r_val(r,c);
    %fprintf('accumulated_sum: %d\n',sum_r);
    sum_g = sum_g + g_val(r,c);
    sum_b = sum_b + b_val(r,c); 
    
    %fprintf('sum_r: %d\n',sum_r);
    %fprintf('sum_g: %d\n',sum_g);
    %fprintf('sum_b: %d\n',sum_b);
    
    %LAB Pixels
    sum_lab_l = sum_lab_l + lab_l(r,c);
    sum_lab_a = sum_lab_a + lab_a(r,c);
    sum_lab_b = sum_lab_b + lab_b(r,c);
    
    %YCbCr Pixels
    sum_ycbcr_y  = sum_ycbcr_y  +  y_val(r,c);
    sum_ycbcr_cb = sum_ycbcr_cb + cb_val(r,c);
    sum_ycbcr_cr = sum_ycbcr_cr + cr_val(r,c);
    
    %grayImae Pixels
    sum_gray = sum_gray + gray_img(r,c);

    
end


mean_r = (sum_r) / (sum_r + sum_g + sum_b); 
mean_g = (sum_g) / (sum_r + sum_g + sum_b); 
mean_b = (sum_b) / (sum_r + sum_g + sum_b); 

mean_lab_l = (sum_lab_l) / (sum_lab_l + sum_lab_a + sum_lab_b);
mean_lab_a = (sum_lab_a) / (sum_lab_l + sum_lab_a + sum_lab_b);
mean_lab_b = (sum_lab_b) / (sum_lab_l + sum_lab_a + sum_lab_b);

mean_ycbcr_y  = (sum_ycbcr_y)  / (sum_ycbcr_y+sum_ycbcr_cb+sum_ycbcr_cr);
mean_ycbcr_cb = (sum_ycbcr_cb) / (sum_ycbcr_y+sum_ycbcr_cb+sum_ycbcr_cr);
mean_ycbcr_cr = (sum_ycbcr_cr) / (sum_ycbcr_y+sum_ycbcr_cb+sum_ycbcr_cr);

mean_gray = sum_gray / scalar_totpixels;
%fprintf('tot: %d\n',scalar_totpixels);


%For Standard-Deviation
for i = 1:scalar_totpixels
    r = pixel_list(i,1);%row for current pixel in image matrix
    c = pixel_list(i,2);%col for current pixel in image matrix
     
    
    %RGB Pixels
    std_rgb_r = std_rgb_r +  power(mean_r - r_val(r,c) , 2);
    std_rgb_g = std_rgb_g +  power(mean_g - g_val(r,c) , 2);
    std_rgb_b = std_rgb_b +  power(mean_b - b_val(r,c) , 2); 


    %LAB Pixels
    std_lab_l = std_lab_l + power(mean_lab_l -lab_l(r,c) , 2);
    std_lab_a = std_lab_a + power(mean_lab_a -lab_a(r,c) , 2);
    std_lab_b = std_lab_b + power(mean_lab_b -lab_b(r,c) , 2);
    
    %YCbCr Pixels
    std_ycrcb_y  = std_ycrcb_y  +  power(mean_ycbcr_y - y_val(r,c),2);
    std_ycrcb_cr = std_ycrcb_cr +  power(mean_ycbcr_cb - cb_val(r,c),2);
    std_ycrcb_cb = std_ycrcb_cb +  power(mean_ycbcr_cr - cr_val(r,c),2);
    
    %grayImae Pixels
    std_gr = std_gr + power(mean_gray-gray_img(r,c),2);
    
end

std_rgb_r = sqrt(std_rgb_r / scalar_totpixels);
std_rgb_g = sqrt(std_rgb_g / scalar_totpixels);
std_rgb_b = sqrt(std_rgb_b / scalar_totpixels);

std_lab_l = sqrt(std_lab_l / scalar_totpixels);
std_lab_a = sqrt(std_lab_a / scalar_totpixels);
std_lab_b = sqrt(std_lab_b / scalar_totpixels);

std_ycrcb_y = sqrt(std_ycrcb_y / scalar_totpixels);
std_ycrcb_cr = sqrt(std_ycrcb_cr / scalar_totpixels);
std_ycrcb_cb = sqrt(std_ycrcb_cb / scalar_totpixels);

std_gr = sqrt(std_gr / scalar_totpixels);

%Skewness and Kurtosis
for i = 1:scalar_totpixels
    r = pixel_list(i,1);%row for current pixel in image matrix
    c = pixel_list(i,2);%col for current pixel in image matrix
    
    %Shape-Feature-b
    sf_b = sf_b + ( (r-stats.Centroid(1)) * (c-stats.Centroid(2)) ) ; 
    
    %Skewness of RGB Pixels
    sk_rgb_r = sk_rgb_r +  power( r_val(r,c) - mean_r , 3);
    sk_rgb_g = sk_rgb_g +  power( g_val(r,c) - mean_g , 3);
    sk_rgb_b = sk_rgb_b +  power( b_val(r,c) - mean_b,  3);
    
    %Kurtosis of RGB Pixels
    kur_rgb_r = kur_rgb_r +  power( r_val(r,c) - mean_r , 4);
    kur_rgb_g = kur_rgb_g +  power( g_val(r,c) - mean_g , 4);
    kur_rgb_b = kur_rgb_b +  power( b_val(r,c) - mean_b , 4); 
    
    %Skewness of LAB Pixels
    sk_lab_l = sk_lab_l + power( lab_l(r,c) - mean_lab_l , 3);
    sk_lab_a = sk_lab_a + power( lab_a(r,c) - mean_lab_a , 3);
    sk_lab_b = sk_lab_b + power( lab_b(r,c) - mean_lab_b , 3);
    
    %Kurtosis of LAB Pixels
    kur_lab_l = kur_lab_l + power( lab_l(r,c) - mean_lab_l , 4);
    kur_lab_a = kur_lab_a + power( lab_a(r,c) - mean_lab_a , 4);
    kur_lab_b = kur_lab_b + power( lab_b(r,c) - mean_lab_b , 4);
    
    %Skewness of YCbCr Pixels
    sk_ycrcb_y  = sk_ycrcb_y  +  power( y_val(r,c) -  mean_ycbcr_y , 3);
    sk_ycrcb_cr = sk_ycrcb_cr +  power(cb_val(r,c) - mean_ycbcr_cb , 3);
    sk_ycrcb_cb = sk_ycrcb_cb +  power(cr_val(r,c) - mean_ycbcr_cr , 3);
    
    %Kurtosis of YCbCr Pixels
    kur_ycrcb_y  = kur_ycrcb_y  +  power( y_val(r,c) -  mean_ycbcr_y , 4);
    kur_ycrcb_cr = kur_ycrcb_cr +  power(cb_val(r,c) - mean_ycbcr_cb , 4);
    kur_ycrcb_cb = kur_ycrcb_cb +  power(cr_val(r,c) - mean_ycbcr_cr , 4);
    
    %Skewness of grayImage Pixels
    sk_gr = sk_gr + power(gray_img(r,c) - mean_gray , 3);
    
    %Kurtosis of grayImage Pixels
    kur_gr = kur_gr + power(gray_img(r,c) - mean_gray , 4); 
end        

%Skewness Calculation for RGB
sk_rgb_r = sk_rgb_r / ( (scalar_totpixels - 1) * power(std_rgb_r,3) );
sk_rgb_g = sk_rgb_g / ( (scalar_totpixels - 1) * power(std_rgb_g,3) );
sk_rgb_b = sk_rgb_b / ( (scalar_totpixels - 1) * power(std_rgb_b,3) );

%Skewness Calculation for LAB
sk_lab_l = sk_lab_l / ( (scalar_totpixels - 1) * power(std_lab_l,3) );
sk_lab_a = sk_lab_a / ( (scalar_totpixels - 1) * power(std_lab_a,3) );
sk_lab_b = sk_lab_b / ( (scalar_totpixels - 1) * power(std_lab_b,3) );

%Skewness Calculation for YCrCB
sk_ycrcb_y = sk_ycrcb_y   / ( (scalar_totpixels - 1) * power(std_ycrcb_y,3) );
sk_ycrcb_cr = sk_ycrcb_cr / ( (scalar_totpixels - 1) * power(std_ycrcb_cr,3) );
sk_ycrcb_cb = sk_ycrcb_cb / ( (scalar_totpixels - 1) * power(std_ycrcb_cb,3) );

%Skewness Calculation for Gray
sk_gr = sk_gr / ( (scalar_totpixels - 1) * power(std_gr,3) );

%Kurtosis Calculation for RGB
kur_rgb_r = kur_rgb_r / ( (scalar_totpixels - 1) * power(std_rgb_r,4) );
kur_rgb_g = kur_rgb_g / ( (scalar_totpixels - 1) * power(std_rgb_g,4) );
kur_rgb_b = kur_rgb_b / ( (scalar_totpixels - 1) * power(std_rgb_b,4) );

%Kurtosis Calculation for LAB
kur_lab_l = kur_lab_l / ( (scalar_totpixels - 1) * power(std_lab_l,4) );
kur_lab_a = kur_lab_a / ( (scalar_totpixels - 1) * power(std_lab_a,4) );
kur_lab_b = kur_lab_b / ( (scalar_totpixels - 1) * power(std_lab_b,4) );

%Kurtosis Calculation for YCrCB
kur_ycrcb_y =  kur_ycrcb_y  / ( (scalar_totpixels - 1) * power(std_ycrcb_y,4) );
kur_ycrcb_cr = kur_ycrcb_cr / ( (scalar_totpixels - 1) * power(std_ycrcb_cr,4) );
kur_ycrcb_cb = kur_ycrcb_cb / ( (scalar_totpixels - 1) * power(std_ycrcb_cb,4) );

%Kurtosis Calculation for Gray
kur_gr = kur_gr / ( (scalar_totpixels - 1) * power(std_gr,4) );

mean_vector = [mean_r,mean_g,mean_b,mean_lab_l,mean_lab_a,mean_lab_b,mean_ycbcr_y,mean_ycbcr_cb,mean_ycbcr_cr,mean_gray];
std_vector = [std_rgb_r,std_rgb_g,std_rgb_b,std_lab_l,std_lab_a,std_lab_b,std_ycrcb_y,std_ycrcb_cr,std_ycrcb_cb,std_gr];
sk_vector = [sk_rgb_r,sk_rgb_g,sk_rgb_b,sk_lab_l,sk_lab_a,sk_lab_b,sk_ycrcb_y,sk_ycrcb_cr,sk_ycrcb_cb,sk_gr];
kur_vector = [kur_rgb_r,kur_rgb_g,kur_rgb_b,kur_lab_l,kur_lab_a,kur_lab_b,kur_ycrcb_y,kur_ycrcb_cr,kur_ycrcb_cb,kur_gr];

%shape features
sf_a = scalar_totpixels / (stats.Perimeter).^2; %Shape-Feature-a
% sf_b: Shape feature-b is calculated in loop
sf_c = scalar_totpixels/ stats.Area;
shape_feature = [sf_a,sf_b,sf_c];

%Location Features
lf_a = (stats.Centroid(1) - image_centre(1));
lf_b = (stats.Centroid(2) - image_centre(2));
lf_c = sqrt( (lf_a).^2 + (lf_b).^2 ) ;
location_feature = [lf_a,lf_b,lf_c];


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
 
 gray_img = rgb2gray(img);
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
 
 %filter params
 [s1,s2]=size(image_data);
 sigma = 0.5;
 h = fspecial('gaussian',[s1 s2],sigma);
 fil_val = filter_operation( image_data,h )
 


feature_vector = [mean_vector;std_vector;sk_vector;kur_vector];



%end
