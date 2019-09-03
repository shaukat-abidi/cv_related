clear all
close all
clc
%[xSphere,ySphere,zSphere] = sphere(16);          %# Points on a sphere
%scatter3(xSphere(:),ySphere(:),zSphere(:),'.');  %# Plot the points
%axis equal;   %# Make the axes scales match
%hold on;      %# Add to the plot

img_rgb = imread('original_rgb_image.jpg');     %# Load a sample image
superpixels_outlined = imread('outlined_superpixels.jpg');
superpixels_colored = imread('colored_superpixels.jpg');
xImage = [-0.5 0.5; -0.5 0.5];   %# The x data for the image corners
yImage = [1 1;-1 -1];             %# The y data for the image corners
zImage = [-0.1 -0.1; -0.1 -0.1];   %# The z data for the image corners
surf(xImage,yImage,zImage,...    %# Plot the surface
     'CData',img_rgb,...
     'FaceColor','texturemap');
 
hold on

%xImage = [-0.5 0.5; -0.5 0.5];   %# The x data for the image corners
%yImage = [0 0;0 0];             %# The y data for the image corners
%zImage = [0.5 0.5; -0.5 -0.5];   %# The z data for the image corners
xImage = [-0.5 0.5; -0.5 0.5];   %# The x data for the image corners
yImage = [1 1;-1 -1];             %# The y data for the image corners
zImage = [-0.06 -0.06;-0.06 -0.06];   %# The z data for the image corners
surf(xImage,yImage,zImage,...    %# Plot the surface
     'CData',superpixels_outlined,...
     'FaceColor','texturemap');

xImage = [-0.5 0.5; -0.5 0.5];   %# The x data for the image corners
yImage = [1 1;-1 -1];             %# The y data for the image corners
zImage = [0 0;0 0];   %# The z data for the image corners
surf(xImage,yImage,zImage,...    %# Plot the surface
     'CData',superpixels_colored,...
     'FaceColor','texturemap');
 
 xlabel('x');
ylabel('y');
zlabel('z');