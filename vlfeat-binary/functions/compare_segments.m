function compare_segments(rgb_img,segment_list,seg_no,image_pixel_list,histogram_vectors)
%visualise segments in different windows
%It qualitatively compares segments from over-segmentation algorithm by the segments
%obtained through the dense sampling of SIFTS
%org_ipl: original image pixel list obtained from txt file of
%over-segmentation algorithm
%histogram_vectors: contains processed image segments
%processed_ipl: processed image pixel list
[pixel_list] = get_seg_pl(segment_list,seg_no,image_pixel_list);
view_segments(rgb_img,pixel_list)
view_segments(rgb_img,histogram_vectors(seg_no).pixels)

end

