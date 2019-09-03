function [ output_val ] = filter_operation( A,B )
% A-very naive implementation of convolving A and B
%Actually we are multiplying every respective element of A and B 
%and then summing it up to get output_value
C = A .* B ; % element-wise multiplication
output_val = sum(C(:));

end

