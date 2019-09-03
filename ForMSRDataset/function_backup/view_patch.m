function view_patch(img,pixel_list)
%view segments
close all
imshow(img)
hold on
scatter(pixel_list(:,1),pixel_list(:,2),'r');
hold off
end