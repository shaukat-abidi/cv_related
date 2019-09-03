function [ seg_no ] = get_segId( seg_ids , pixel_index, req_pixel_index )
% seg_ids : vector containing segment ids (Nx1)
% pixel_index: [p_x,p_y] where pixel ids are stored in N rows 
% The i^th row in pixel_index has following segment id -- seg_ids(i)    
% req_pixel_index: required pixel index whose segment id needs to be found 
% seg_no : our needed seg_no

[row,col]=size(pixel_index);

seg_no=-1;

for i=1:row
    if( isequal(pixel_index(i,:) , req_pixel_index ) )
        seg_no = seg_ids(i);
        break;
    else
        continue;
    end    
end

end

