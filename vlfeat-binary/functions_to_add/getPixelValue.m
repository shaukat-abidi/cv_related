function [r,g,b] = getPixelValue(img,index)
x=index(2);
y=index(1);
r=img(x,y,1);
g=img(x,y,2);
b=img(x,y,3);
end