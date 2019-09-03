function [ score ] = sigmoid( theta,offset,x )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
score = 1./ ( 1 + exp( -1 * (theta' * x + offset) ) );
end

