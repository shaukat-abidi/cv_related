function [ fid ] = find_index( string_array,str_to_comp )
%FIND_INDEX Summary of this function goes here
%   Detailed explanation goes here
total_strings = length(string_array);
fid=-1;
for i=1:total_strings
    if ( strcmp(string_array(i,:),str_to_comp) == 1 )
        fid=i;
        break
    end
end

end

