function [ output_image ] = gen_binary_image( input_image,pixel_list )
% gen_binary_image: Returns binary image of "input_image" with 1's over
% "pixel_list"

%Rows and Cols for image
m=size(input_image,1);
n=size(input_image,2);

%Generate matrix with zeroes
output_image = zeros(m,n);

%filling those pixels with ones that are stored in "pixel_list"
output_image( pixel_list(:,2),pixel_list(:,1) ) = 1;

end

