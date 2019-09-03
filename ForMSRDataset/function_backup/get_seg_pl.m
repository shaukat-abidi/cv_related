function [pixel_list] = get_seg_pl(all_segment_vector,s_no,p)
%get_seg_pl: get segment's pixel list
%returns all pixels forming segment#=s_no
%Let assume there are 1,...,C segments and 1,...,N pixels
%all_segment_vector: (Nx1) vector relating pixel number to valid segment number between [1,C] 
%s_no is scalar between [1,C]
%p is Nx2 vectors containing pixel index
temp = find(all_segment_vector == s_no);
pixel_list = p(temp,:);
end