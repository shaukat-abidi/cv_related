function view_segments(img,histogram_vectors,total_no_segments)
%view segments
imshow(img)
hold on 
for iter_segment = 1:total_no_segments
    r=0;
    g=0;
    b=0;
    
        if (histogram_vectors(iter_segment).label == 1)
            r=128;
            g=0;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 2)
            r=0;
            g=128;
            b=0;
        end

        if(histogram_vectors(iter_segment).label == 3)
            r=128;
            g=128;
            b=0;
        end

        if(histogram_vectors(iter_segment).label == 4)
            r=0;
            g=0;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 5)
            r=128;
            g=0;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 6)
            r=0;
            g=128;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 7)
            r=128;
            g=128;  
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 8)
            r=64;
            g=0;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 9)
            r=192;
            g=0;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 10)
            r=64;
            g=128;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 11)
            r=192;
            g=128;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 12)
            r=64;
            g=0;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 13)
             r=192;
             g=0;
             b=128;
        end

        if (histogram_vectors(iter_segment).label == 14)
            r=64;
            g=128;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 15)
            r=192;
            g=128;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 16)
            r=0;
            g=64;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 17)
            r=128;
            g=64;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 18)
            r=0;
            g=192;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 19)
            r=128;
            g=64;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 20)
            r=0;
            g=192;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 21)
            r=128;
            g=192;
            b=128;
        end

        if (histogram_vectors(iter_segment).label == 22)
            r=64;
            g=64;
            b=0;
        end

        if (histogram_vectors(iter_segment).label == 23)
            r=192;
            g=64;
            b=0;
        end
        
        if(histogram_vectors(iter_segment).id ~= -1) % && histogram_vectors(iter_segment).label == 12 )
            pixel_list = histogram_vectors(iter_segment).pixels;
            x=pixel_list(:,1);
            y=pixel_list(:,2);
            r=r./(r+g+b);
            g=g./(r+g+b);
            b=b./(r+g+b);
            scatter(x,y,5,[r,g,b]);
        end
end
        hold off
end