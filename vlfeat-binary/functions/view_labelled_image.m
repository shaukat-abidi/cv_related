function view_labelled_image(rgb_image,histogram_vectors,total_no_segments,filename )
save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForMSRDataset/intermediate_results/';
m=size(rgb_image,1);
n=size(rgb_image,2);
matrix_r = zeros(m,n);
matrix_g = zeros(m,n);
matrix_b = zeros(m,n);

	for iter_segments=1:total_no_segments
		current_label = histogram_vectors(iter_segments).valid_object_id;
		[r,g,b] = get_color_using_label(current_label);
		r = uint8(r);
		g = uint8(g);
		b = uint8(b);
		
		if(histogram_vectors(iter_segments).no_feats ~= 0)
			for iter_pixels=1:size(histogram_vectors(iter_segments).pixels , 1)
				current_pixel = histogram_vectors(iter_segments).pixels(iter_pixels,:);
				matrix_r(current_pixel(2),current_pixel(1)) = r;
				matrix_g(current_pixel(2),current_pixel(1)) = g;
				matrix_b(current_pixel(2),current_pixel(1)) = b;
				 
			end
		end
	end
	generated_image = cat(3,matrix_r,matrix_g,matrix_b);
	save_filename = strcat(save_path,filename);
    imwrite(generated_image,save_filename,'png');
	%figure,imshow(generated_image)
end
