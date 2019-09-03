function generated_image = gen_segmented_image(rgb_image,histogram_vectors,total_no_segments,filename )
save_path = '/home/ssabidi/Shaukat/vlfeat-binary/toolbox/staticAR/intermediate_results/';
m=size(rgb_image,1);
n=size(rgb_image,2);
matrix_r = uint8(255 * ones(m,n));
matrix_g = uint8(255 * ones(m,n));
matrix_b = uint8(255 * ones(m,n));
%total_pixels_written = 0;
	for iter_segments=1:total_no_segments
	    if(histogram_vectors(iter_segments).valid_segment == 1)
			for iter_pixels=1:size(histogram_vectors(iter_segments).pixels , 1)
				current_pixel = histogram_vectors(iter_segments).pixels(iter_pixels,:);
				matrix_r(current_pixel(2),current_pixel(1)) = rgb_image(current_pixel(2),current_pixel(1),1);
				matrix_g(current_pixel(2),current_pixel(1)) = rgb_image(current_pixel(2),current_pixel(1),2);
				matrix_b(current_pixel(2),current_pixel(1)) = rgb_image(current_pixel(2),current_pixel(1),3);
				%total_pixels_written = total_pixels_written + 1; 
			end
		end
	end
	generated_image = cat(3,matrix_r,matrix_g,matrix_b);
	save_filename = strcat(save_path,filename);
    imwrite(generated_image,save_filename,'png');
    %total_pixels_written
	%figure,imshow(generated_image)
end
