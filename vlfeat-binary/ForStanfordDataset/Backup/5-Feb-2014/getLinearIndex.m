function linear_index = getLinearIndex(x,y,width,height)
%This is width and height of MSR21-Images
%width = 320;
%height = 213;
linear_index = (y-1)*width + x;
end