function object_id = getGroundTruth(r,g,b)
    
object_id = -1;

if (r==128 && g==0 && b==0)
    object_id = 1;
end

if (r==0 && g==128 && b==0)
    object_id = 2;
end

if (r==128 && g==128 && b==0)
    object_id = 3;
end

if (r==0 && g==0 && b==128)
    object_id = 4;
end

if (r==128 && g==0 && b==128)
    object_id = 5;
end

if (r==0 && g==128 && b==128)
    object_id = 6;
end

if (r==128 && g==128 && b==128)
    object_id = 7;
end

if (r==64 && g==0 && b==0)
    object_id = 8;
end

if (r==192 && g==0 && b==0)
    object_id = 9;
end

if (r==64 && g==128 && b==0)
    object_id = 10;
end

if (r==192 && g==128 && b==0)
    object_id = 11;
end

if (r==64 && g==0 && b==128)
    object_id = 12;
end

if (r==192 && g==0 && b==128)
    object_id = 13;
end

if (r==64 && g==128 && b==128)
    object_id = 14;
end

if (r==192 && g==128 && b==128)
    object_id = 15;
end

if (r==0 && g==64 && b==0)
    object_id = 16;
end

if (r==128 && g==64 && b==0)
    object_id = 17;
end

if (r==0 && g==192 && b==0)
    object_id = 18;
end

if (r==128 && g==64 && b==128)
    object_id = 19;
end

if (r==0 && g==192 && b==128)
    object_id = 20;
end

if (r==128 && g==192 && b==128)
    object_id = 21;
end

if (r==64 && g==64 && b==0)
    object_id = 22;
end

if (r==192 && g==64 && b==0)
    object_id = 23;
end

end