%% Simulating PDE with Fourier Spectral method
% Note: Spectral method in Fourier domain does not require any boundary
% condition, as it is handled during the derivative approximation via
% wave-numbers on the Fourier-modes.

clear;close all;clc

Lx=20.0; Ly=30.0;                                       % Dimensions of the plate
Nx=41; Ny=61;                                         % Number of discretisation points
[X,Y] = meshgrid(linspace(0,Lx,Nx),linspace(0,Ly,Ny)); % (x,y)-grid 

% Physical parameters 
Kappa = 148; rho = 2328; cp = 700; 
Alpha = Kappa/(rho*cp);

% Wave number for the Fourier Mode
k=(2*pi/Lx)*[0:(Nx-1)/2 -(Nx-1)/2:-1].'; % Wave-number for x
l=(2*pi/Ly)*[0:(Ny-1)/2 -(Ny-1)/2:-1].'; % Wave-number for y
k(1) = 1e-6; l(1) = 1e-6;                % Removing numerical stiffness
[KX,KY] = meshgrid(k,l);                 % Meshing the wave-numbers into (x,y)-grid
KXY = KX.^2+KY.^2;                       % Laplacian derivative in Fourier domain
KXY_reshaped = reshape(KXY,Nx*Ny,1);     % Reshaping for odeXX routines

% Initial conditions
T0=35;                                          % Ambient Temperature
% T = T0+zeros(Ny,Nx);
% T = T0+cos(X - Lx/2).*cos(Y - Ly/2);
T = T0+sin((pi*X)/Lx).*sin((pi*Y)/Ly);          % So far, the best resulting one
% T = T0+exp(-((X/Lx-1/2)).^2-((Y/Ly-1/2)).^2);
Tt_reshaped=reshape(fft2(T),Nx*Ny,1);           % Reshaping for odeXX routines

% Source(s) term
Tsource = 2000000;                         % Heat flux (in W/m^2)
Sx1 = round(0.25*Nx); Sy1 = round(0.5*Ny); % Location of heat flux 1
Sx2 = round(0.75*Nx); Sy2 = round(0.5*Ny); % Location of heat flux 2

Source = zeros(Ny,Nx);                      % Pre-allocating the Source term
Source(Sy1,Sx1) = Tsource/rho/cp;           % u1(t, x, y)
Source(Sy2,Sx2) = Tsource/rho/cp;           % u2(t, x, y)
Source_fft = reshape(fft2(Source),Nx*Ny,1); % Reshaping for odeXX routines

% Solving the PDE
dt = 0.05;        % Time step
tspan = 0:dt:600; % Time span 
[t,Ttsol] = ode15s('heatprop2d_rhs',tspan, Tt_reshaped,[], ...
                    Alpha, KXY_reshaped, Source_fft);   % Solving for the T(t,x,y) in Fourier domain 
%% Simulating the calculated T

for j = 1:length(t)
    Tsol_plot=real(ifft2(reshape(Ttsol(j,:),Ny,Nx)));
    surfc(X,Y,Tsol_plot), shading interp; colorbar, % view(2); 
    title(sprintf("Time = %.3f seconds", t(j)));
    hold on
    drawnow
    pause(0.01)
end

% For documenting at certain timestamps, the code below can be
% uncommented.

% for j = 201
%     Tsol_plot=real(ifft2(reshape(Ttsol(j,:),Ny,Nx)));
%     figure(1); surfc(X,Y,Tsol_plot), shading interp; % colorbar, % view(2); 
%     title(sprintf("Time = %.3f seconds", t(j)));
%     set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold',  'LineWidth', 2);
%     drawnow
%     pause(0.01)
% end
% 
% for j = 1001
%     Tsol_plot=real(ifft2(reshape(Ttsol(j,:),Ny,Nx)));
%     figure(2); surfc(X,Y,Tsol_plot), shading interp; % colorbar, % view(2); 
%     title(sprintf("Time = %.3f seconds", t(j)));
%     set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold',  'LineWidth', 2);
%     drawnow
%     pause(0.01)
% end
% 
% for j = 12001
%     Tsol_plot=real(ifft2(reshape(Ttsol(j,:),Ny,Nx)));
%     figure(3); surfc(X,Y,Tsol_plot), shading interp; % colorbar, % view(2); 
%     title(sprintf("Time = %.3f seconds", t(j)));
%     set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold',  'LineWidth', 2);
%     drawnow
%     pause(0.01)
% end

%% Creating Snapshot Matrix (U_snap)

% Capturing simulation data (aka the raw snapshots)
Tsnap = zeros(Ny,Nx,length(t));
for j = 1:length(t) 
    Tsnap(:,:,j) = real(ifft2(reshape(Ttsol(j,:),Ny,Nx)));
end

% Creating the (useable) snapshot matrix  U_snap
Nt = length(Tsnap(1,1,:));
Tsnap = reshape(permute(Tsnap, [3 2 1]), Nt, [ ]); % Reshape data into a matrix S with Nt rows
U_snap = Tsnap - repmat(mean(Tsnap,1), Nt, 1); % Subtract the temporal mean from each row
%% Note 2: POD methods
% Here, one can choose to compute the POD with the direct method or the SVD
% method. Just uncomment one of each and comment the other one. On par, the
% direct POD method is generally faster than the SVD one.

%% 1. Snapshot POD
% C = (U_snap'*U_snap)/(Nt-1); % Create covariance matrix
% 
% [PHI, LAM] = eig(C,'vector'); % Solve eigenvalue problem
% 
% % Sort eigenvalues and eigenvectors
% [lambda,ilam] = sort(LAM,'descend');
% figure(4);scatter(1:length(lambda),lambda./sum(lambda),'k'); title("Singular Values") % Plotting the Singular Values
% 
% PHI = PHI(:, ilam); % These are the spatial modes
% A = U_snap*PHI; % Calculate time coefficients
% 

%% 2. SVD POD method
[L,SIG,R] = svd(U_snap/sqrt(Nt-1),"econ","vector"); 
PHI = R;                                            % PHI are the spatial modes
A = U_snap*PHI;                                     % A is the time coefficients
figure(4);scatter(1:length(SIG),SIG/sum(SIG)*100,40,'b',"filled"); 
title("Singular Values", 'fontsize', 15,'fontweight','bold') % Plotting the Singular Values
set(gca,'FontName','Arial','FontSize',14,'FontWeight','Bold',  'LineWidth', 2);
ylabel("Total Energy [%]", "fontsize", 14, "fontweight","bold");

% Semilogy-styled singular value plot
figure(5);semilogy(1:length(SIG),SIG/sum(SIG),'ob'); title("Singular Values");
set(gca,'FontName','Arial','FontSize',14,'FontWeight','Bold',  'LineWidth', 2);
ylabel("Total Energy", "fontsize", 14, "fontweight","bold");
%% Model Reduction of k-th approximation order

k = 5;                           % k-spatial modes
Utilde_k = A(:,1:k)*PHI(:,1:k)'; % Reconstruction on mode k

%% Plotting the POD results

for j=1:length(t)
    Utilde_k_plot = reshape(Utilde_k(j,:),Nx,Ny);
    figure("NumberTitle","off","Name","POD Simulation");
    surfc(X,Y,Utilde_k_plot.'), shading interp; colorbar %, view(2); 
    title(sprintf("Time = %.3f seconds", t(j)));
    hold on;
    drawnow
    pause(0.01)
end

% For documenting at certain timestamps, the code below can be
% uncommented.
% for j = 201
%     Utilde_k_plot = reshape(Utilde_k(j,:),Nx,Ny);
%     figure(6)
%     surfc(X,Y,T0+Utilde_k_plot.'), shading interp; % colorbar, % view(2); 
%     title(sprintf("POD Time = %.3f seconds", t(j)));
%     set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold',  'LineWidth', 2);
%     drawnow
%     pause(0.01)
% end
% 
% for j = 1001
%     Utilde_k_plot = reshape(Utilde_k(j,:),Nx,Ny);
%     figure(7)
%     surfc(X,Y,T0+Utilde_k_plot.'), shading interp; % colorbar, % view(2); 
%     title(sprintf("POD Time = %.3f seconds", t(j)));
%     set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold',  'LineWidth', 2);
%     drawnow
%     pause(0.01)
% end
% 
% for j = 12001
%     Utilde_k_plot = reshape(Utilde_k(j,:),Nx,Ny);
%     figure(8)
%     surfc(X,Y,T0+Utilde_k_plot.'), shading interp; % colorbar, % view(2); 
%     title(sprintf("POD Time = %.3f seconds", t(j)));
%     set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold',  'LineWidth', 2);
%     drawnow
%     pause(0.01)
% end