function [STATS]= superpixel_props(binary_image)
%return superpixels properties
STATS = regionprops(binary_image, 'Area', 'BoundingBox', 'Centroid', 'PixelList','Perimeter' );
end