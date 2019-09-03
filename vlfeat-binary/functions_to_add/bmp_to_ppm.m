clear all,close all,clc

for i=1:21
b=int2str(i);
a=strcat('20_',b,'_s.bmp');
img=imread(a);
b=strcat('20_',b,'_s.ppm');
imwrite(img,b);
clear all
end
