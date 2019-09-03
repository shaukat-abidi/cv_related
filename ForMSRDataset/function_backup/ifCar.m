function decision = ifCar(r,g,b,ground_truth)
decision = 0;
    if (r==ground_truth(1)&& g==ground_truth(2) && b==ground_truth(3))
        decision = 1; %This pixel belongs to car
    end
end