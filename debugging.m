point_world = [10;10;10];
point_world_aug = [point_world; 1];
t = [10; 10; 10]; % translate
R = [ 6.68964732e-01, -7.43294146e-01,  0.00000000e+00;
     7.43294140e-01,  6.68964726e-01, -1.34522787e-04;
     9.99900001e-05,  8.99910001e-05,  9.99999991e-01];

E = [R, t];

point_camera = E * point_world_aug