function [ r_seg,p,A,total_segments ] = readFile( filename )
% s_no : segment no obtained from Felzenschwalbs algorithm
% p: [p_x,p_y]
% tot_segments: total segments in image
% r_seg : reordered segments ids from 1 to tot_segments
% A: matrix similar to the "filename". It contains [s_no,px,py]
% total_segments: total number of segments in image obtained from
% "filename"'s first line


fid = fopen(filename, 'r');
theSize = sscanf(fgets(fid), '%d')'; 
rows = theSize(1);
cols = theSize(2);
total_segments  = theSize(3);
A=zeros(rows*cols,3);

for i=0:cols-1
    for j=0:rows-1
        element_no = i*rows + j; %equation to convert image matrix in separate elements  
        A(element_no+1,:)=sscanf(fgets(fid),'%d');
    end
end
s_no = A(:,1);
r_seg = s_no;
p = A(:,2:3);

%Refine segment numbers because segment ids doesn't 
%start from 1
seg_ids = unique(s_no);

for i=1:size(seg_ids,1)
    r_seg( find( r_seg == seg_ids(i) ) ) =  i;
end

if(total_segments ~= size(seg_ids,1))
    total_segments = -1;
    fprintf('something is wrong. Segments counting failed \n')
end

%for debugging
%[s_no,r_seg]

fclose(fid);
end

