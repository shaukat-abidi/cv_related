clear all,clc;

% To read file
[ s_no,p,A,total_segments ] = readFile('1_1_s_out.txt');

% To get the segment of pixel "query"
query = [113,146];
seg_no = get_segId(s_no,p,query);