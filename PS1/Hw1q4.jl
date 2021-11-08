## Packages
using Roots, Calculus, Plots

## Function for finding root
function findZero(g, init_guess)
    
    xn1 = init_guess;  
    tol = 10^(-8);
    distance = 1;
    iter = 1;

    while distance >= tol
        xn = xn1;
        xn1 = xn - g(xn) / derivative(g, xn);
        xn1 = max(xn1, 0.0001);

        distance = abs(xn1 - xn);
        iter = iter + 1;
    end

    return xn1
end

## Function for defining utility function
function u(c)
    if c <= 0
        u = -Inf;
    else
        u = c^(1/2);
    end
    return u
end

## Parameters
β = 0.8;
πs1 = 1/2;
πs2 = 1/2;
Ys1 = 1;
Ys2 = 1/2;
M = 20; # Number of grids

## Grid for v
vmax = (πs1 * u(Ys1) + πs2 * u(Ys2)) / (1 - β);
vgrid = range(0.0, stop = vmax, length = M);

## Iterate on Value function
Pi1 = zeros(M);
cs1_grid = range(0, stop = Ys1, length = M);
ws1_grid = vgrid;
ws2_grid = vgrid;

distance = 1;
tol = 10^(-8);
iter = 1;

policy_fun = Array{CartesianIndex, 1}(undef, M);
policy_cs1 = Array{Int64, 1}(undef, M);

while distance >= tol
    println("This is iteration $iter")
    Pi = copy(Pi1); 
    
    for vIndex in 1:M
        U = Array{Float64, 3}(undef, M, M, M)
        v = vgrid[vIndex];
        for cs1Index in 1:M, ws1Index in 1:M, ws2Index in 1:M
            cs1 = cs1_grid[cs1Index];
            ws1 = ws1_grid[ws1Index];
            ws2 = ws2_grid[ws2Index];

            # Solve for cs2
            f(cs2) = v - (πs1 * (cs1^(1/2) + β * ws1)) - (πs2 * (cs2^(1/2) + β * ws2));
            cs2 = findZero(f, 1.0);

            U_temp = (πs1 * (u(Ys1 - cs1) + β * Pi[ws1Index])) + 
                     (πs2 * (u(Ys2 - cs2) + β * Pi[ws2Index]));

            if U_temp <= 0.0
                U[cs1Index, ws1Index, ws2Index] = 0.0;
            else
                U[cs1Index, ws1Index, ws2Index] = U_temp;
            end
        end
        Pi1[vIndex], policy_fun[vIndex] = findmax(U[:,:,:]);
    end

    distance = maximum(abs.(Pi1 - Pi));
    iter = iter + 1;
end

## A Check
λ = range(0.0, stop = 0.999, length = 100);
u1 = zeros(100);
u2 = zeros(100);
for i in 1:100
    A = (λ[i] / (1 - λ[i]))^2;
    cs1 = Ys1 * (A / (1 + A));
    cs2 = Ys2 * (A / (1 + A));
    u1[i] = (πs1 * u(cs1) + πs2 * u(cs2)) / (1 - β);
    u2[i] = (πs1 * u(Ys1 - cs1) + πs2 * u(Ys2 - cs2)) / (1 - β);
end

plot(vgrid, Pi1, xlabel = "v", ylabel = "P", label = "VFI", legend =:bottomleft)
plot!(u1, u2, label = "Theoretical")