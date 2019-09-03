function view_segments(img,pixel_list)
%view segments with pixels "pixel_list"
%pixel_list: contains pixels of superpixel under consideration. 
figure
imshow(img)
hold on 
scatter(pixel_list(:,1),pixel_list(:,2),'r');
hold off
end
