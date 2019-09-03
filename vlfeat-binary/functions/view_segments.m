function view_segments(img,pixel_list)
%view segments with pixels "pixel_list"
%pixel_list: contains pixels of superpixel under consideration. 
figure
imshow(img)
hold on 
%scatter_plot = scatter(pixel_list(:,1),pixel_list(:,2),'g');
%alpha(0.2)
scatter_plot = scatter(pixel_list(:,1),pixel_list(:,2),2.5,'g','filled');
hold off
end
