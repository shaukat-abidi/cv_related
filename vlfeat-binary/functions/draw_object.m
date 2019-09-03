function draw_object( img,stats )
%Draws bounding box over image
imshow(img);

x=stats.BoundingBox(1);
y=stats.BoundingBox(2);
w=stats.BoundingBox(3);
h=stats.BoundingBox(4);

hold on
rectangle('Position', [x y w h],'FaceColor','r');
scatter(stats.Centroid(1),stats.Centroid(2),'filled');
hold off

end

