clear;close all;clc
tic
% Physical parameters 
Kappa = 148; 
rho = 2328; 
cp = 700;
Alpha = Kappa/(rho*cp);

% Problem Formulation
Lx = 0.20; Ly = 0.30;
Nx = 20; Ny = 30; % Number of points evaluated in the discretised grid
[Dx, x] = cheb_custom(Nx,-Lx/2,Lx/2); % Defining dx and x with Chebyshev Polynomial w.r.t custom L2 space
[Dy, y] = cheb_custom(Ny,-Ly/2,Ly/2); % Defining dy and y with Chebyshev Polynomial w.r.t custom L2 space
% [Dx, x] = cheb(Nx); % Defining dx and x with Chebyshev Polynomial
% [Dy, y] = cheb(Ny); % Defining dy and y with Chebyshev Polynomial

D2x = Dx^2; D2y = Dy^2; % Defining the Chebyshev's 2nd derivative

[X,Y] = meshgrid(x,y); % Creating the (x,y)-grids

% (Neumann) Boundary conditions 
D2x(end,:) = zeros(1,Nx+1); D2x(1,:) = zeros(1,Nx+1);
D2y(end,:) = zeros(1,Ny+1); D2y(1,:) = zeros(1,Ny+1);

% Creating the 2D-Laplacian
Ix = eye(length(x)); Iy = eye(length(y));
L = kron(Ix,D2y)+kron(D2x,Iy); % 2D Laplacian 


% Initial condition (profiles)
Tamb = 35; % Ambient Temperature
Tinit = Tamb + exp(-(X).^2-(Y).^2);
% Tinit = Tamb + sin((X)/Lx).*sin((Y)/Ly); 
Tinit_reshaped = reshape(Tinit, (Nx+1)*(Ny+1), 1);

% Source(s) term
Tsource = 200000;
Sx1 = round(0.25*Nx); Sy1 = round(0.5*Ny); % Location of heat flux 1
Sx2 = round(0.75*Nx); Sy2 = round(0.5*Ny); % Location of heat flux 2

heatSource = zeros(Ny+1,Nx+1); % Preallocation
heatSource(Sy1,Sx1) = Tsource/rho/cp;
heatSource(Sy2,Sx2) = Tsource/rho/cp;
heatSource = reshape(heatSource,(Nx+1)*(Ny+1),1);

% Solving the PDE
tspan = 0:0.1:600;
[t,Tsol] = ode23t('heatprop2D_rhs', tspan, Tinit_reshaped, [], Alpha, L, heatSource);

%% Plotting the simulation results
% for j=1:length(t)
%     Tplot = reshape(Tsol(j,:),Ny+1,Nx+1);
%     surfc(X,Y,Tplot), shading interp; colorbar, % view(2); 
%     title(sprintf("Time = %.3f seconds", t(j)));
%     hold on
%     drawnow
%     pause(0.01)
% end


%% Creating Snapshot Matrix (U_snap)

% Collecting the raw snapshot data
Traw = zeros(Ny+1,Nx+1,length(t));
for j = 1:length(t) 
    Traw(:,:,j) = reshape(Tsol(j,:),Ny+1,Nx+1); 
end

% Processing the raw data into useable snapshot data
Nt = length(Traw(1,1,:));
Traw = reshape(permute(Traw, [3 2 1]), Nt, [ ]); % Reshape data into a matrix S with Nt rows
U_snap = Traw - repmat(mean(Traw,1), Nt, 1); % Subtract the temporal mean from each row
% U_snap is now the useable snapshot for POD

%% 
% Here, one can choose to compute the POD with the direct method or the SVD
% method. Just uncomment one of each and comment the other one. On par, the
% direct POD method is generally faster than the SVD one.

%% 1. Direct POD method
C = (U_snap'*U_snap)/(Nt-1); % Create covariance matrix

[PHI, LAM] = eig(C,'vector'); % Solve eigenvalue problem


[lambda,ilam] = sort(LAM,'descend'); % Sort eigenvalues and eigenvectors
scatter(1:length(lambda),lambda./sum(lambda),'k'); title("Singular Values") % Plotting the Singular Values

PHI = PHI(:, ilam); % These are the spatial modes
A = U_snap*PHI; % Calculate time coefficients

%% 2. SVD POD method
% [L,SIG,R] = svd(U_snap/sqrt(Nt-1),"econ","vector"); 
% PHI = R;        % PHI are the spatial modes
% A = U_snap*PHI; % A is the time coefficients
% 
% % Plotting the Singular Values
% figure(4);scatter(1:length(SIG),SIG/sum(SIG)*100,40,'b'); title("Singular Values"); % Normal plot
% figure(5);semilogy(1:length(SIG),SIG/sum(SIG),'ob'); title("Singular Values");      % Semilogy plot
%% Reconstructing the k-th order reduced order

k = 5; % k-spatial modes
Utilde_k = A(:,1:k)*PHI(:,1:k)'; % Reconstruction on mode k
toc
%% Plotting the POD results

for j=1:length(t)
    Utilde_k_plot = reshape(Utilde_k(j,:),Nx+1,Ny+1);
%     figure("NumberTitle","off","Name","POD Simulation");
    figure(2);
    surfc(X,Y,Tamb+Utilde_k_plot.'), shading interp; colorbar %, view(2); 
    title(sprintf("Time = %.3f seconds", t(j)));
    drawnow
    hold on
    pause(0.01)
end

