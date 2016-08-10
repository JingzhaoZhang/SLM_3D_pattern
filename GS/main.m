clear all;
tag = 'slmFocal_720_600by800_A';


%% Setup params
% All length value has unit meter in this file.
% The 3d region is behind lens after SLM. 

addpath(genpath('minFunc_2012'))

resolutionScale = 20; % The demagnification scale of tubelens and objective. f_tube/f_objective
lambda = 1e-6;  % Wavelength
focal_SLM = 0.2; % focal length of the lens after slm.
psSLM = 20e-6;      % Pixel Size (resolution) at the scattered 3D region
Nx = 800;       % Number of pixels in X direction
Ny = 600;       % Number of pixels in Y direction


psXHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Nx;      % Pixel Size (resolution) at the scattered 3D region
psYHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Ny;      % Pixel Size (resolution) at the scattered 3D region

useGPU = 1;     % Use GPU to accelerate computation. Effective when Nx, Ny is large (e.g. 600*800).


z = [-100 :4: 100] * 1e-6 ;   % Depth level requested in 3D region.
nfocus = 25;                % z(nfocus) denotes the depth of the focal plane.
thresholdh = 20000000;          % Intensity required to activate neuron.
thresholdl = 0;             % Intensity required to not to activate neuron.

%% Point Targets
% radius = 9.9 * 1e-6 ; % Radius around the point.
% targets = [ 0, 0, z(10) * 1e6;   0,0, z(40) * 1e6;] * 1e-6 ; % Points where we want the intensity to be high.
%targets = [ 0, 0, z(25) * 1e6] * 1e-6 ;
% %targets = [150,150,450; 0, 0, 500; -150,-150,550;] * 1e-6 ; % Points where we want the intensity to be high.
% maskfun = @(zi)  generatePointMask( targets, radius, zi, Nx, Ny, psXHolograph,psYHolograph, useGPU);


%% Complex Target
load('largeAB');
ims(:,:,1) = maskA' * 10;
imdepths = z(25);
% zrange1 = [450,480] * 1e-6;
% zrange2 = [550,580] * 1e-6;

%maskfun = @(zi) generateComplexMask( zi, Nx, Ny, maskA', zrange1, maskB', zrange2);



%% Optimization


HStacks = zeros(Nx, Ny, numel(z));
if useGPU
    HStacks = gpuArray(HStacks);
end
for i = 1 : numel(z)
    HStacks(:,:,i) = GenerateFresnelPropagationStack(Nx, Ny, z(i)-z(nfocus), lambda, psXHolograph,psYHolograph, useGPU);
end
kernelfun = @(x) HStacks(:,:,x);
% kernelfun = @(i) GenerateFresnelPropagationStack(Nx, Ny, z(i)-z(nfocus), lambda,psXHolograph,psYHolograph, focal_SLM, useGPU);

%load('gs_simulation.mat')
%load('../data/feasible_points.mat')
%Ividmeas = getFeasiblePointTargets( targets, radius, z, nfocus, resolutionScale, lambda, focal_SLM, psSLM, Nx, Ny );
Ividmeas = getFeasibleComplexTargets( ims, imdepths, z, nfocus, resolutionScale, lambda, focal_SLM, psSLM, Nx, Ny );



intensity = 1/Nx/Ny;
source = sqrt(intensity) * ones(Nx, Ny);
%source = 10000*source1;
im = source;
maxiter = 30;
figure();
for n = 1:maxiter
    for i = 1:numel(z)
        HStack = kernelfun(i);
        imagez = fftshift(fft2(im .* HStack));
        target = Ividmeas(:,:,i) ;
        %target = maskfun(z(i)) * 10;
        imagez = sqrt(target) .* exp(1i * angle(imagez));
        im =  ifft2(ifftshift(imagez))./HStack;
        im = source.*exp(1i * angle(im));
    end
    %im = source.*exp(1i * angle(im));
    if mod(n, 10)==0
        display(n)
        imagesc(angle(im));drawnow;pause(0.1);
    end
end


source1 = source;
phase1 = gather(angle(im));

%%
% load('starTarget.mat')
% load('complextarget')
% source1 = star(1:300,1:300);
% phase1 = mask;


%% plot
Ividmeas = zeros(Nx, Ny, numel(z));
usenoGPU = 0;
figure();
for i = 1:numel(z)
    HStack = GenerateFresnelPropagationStack(Nx,Ny,z(i) - z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
    imagez = fresnelProp(phase1, source1, HStack);
    Ividmeas(:,:,i) = imagez;
    imagesc(imagez);colormap gray;title(sprintf('Distance z %d', z(i)));
    colorbar;
    caxis([0, 200]);
%     filename = sprintf('pointTarget%d.png', i);
%     print(['data/' filename], '-dpng')
    pause(0.1);
end
%save('gs_simulation.mat', 'Ividmeas', 'source1', 'phase1')
%save(['source_phase_result_' tag '.mat'], 'source1', 'phase1', 'source2', 'phase2', 'hologram');

save(['gs_result_' tag '.mat'], 'phase1');

