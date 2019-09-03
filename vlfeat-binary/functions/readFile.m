function [ r_seg,p,A,total_segments,segment_density ] = readFile( filename )
% s_no : segment no obtained from Felzenschwalbs algorithm
% p: [p_x,p_y]
% tot_segments: total segments in image
% r_seg : reordered segments ids from 1 to total_segments
% A: matrix similar to the "filename". It contains [s_no,px,py]
% total_segments: total number of segments in image obtained from
% "filename"'s first line
% segment_density:   (total_no_segments x 2) Column-1 contains segment# and
% Column-2 contains total pixels inside corresponding segment 


fid = fopen(filename, 'r');
theSize = sscanf(fgets(fid), '%d')'; 
rows = theSize(1);
cols = theSize(2);
total_segments  = theSize(3);
A=zeros(rows*cols,3);
segment_density = zeros(total_segments,2);

for i=0:cols-1
    for j=0:rows-1
        element_no = i*rows + j; %equation to convert image matrix in separate elements  
        A(element_no+1,:)=sscanf(fgets(fid),'%d');
    end
end
s_no = A(:,1); %segment# obtained from txt file
r_seg = s_no;  %storing s_no in r_seg so that we can re-order it
p = A(:,2:3);  %storing pixel values separately

%Refine segment numbers because segment ids in s_no doesn't 
%start from 1.
%Finding unique segment# in s_no (in increasing order)
seg_ids = unique(s_no);

for i=1:size(seg_ids,1)
    desired_indices = find( r_seg == seg_ids(i) );
    total_pixels = size(desired_indices,1);
    r_seg(desired_indices) =  i;
    
    %storing re-ordered segment # with its corresponding number of 
    %pixels
    segment_density(i,1) = i; %re-ordered segment number
    segment_density(i,2) = total_pixels; %total number of pixels inside segment# i
    
end

check_value = sum(segment_density(:,2));


if(total_segments ~= size(seg_ids,1) || check_value~= size(p,1) )
    total_segments = -1;
    r_seg = [];
    p = [];
    A = [];
    total_segments=[];
    segment_density = [];
    fprintf('Error readFile(). Segments counting failed. \n')
end

%for debugging
%[s_no,r_seg]

fclose(fid);
end

