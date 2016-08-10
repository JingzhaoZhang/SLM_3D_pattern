%% Setup params
% All length value has unit meter in this file.
% The 3d region is behind lens after SLM. 

resolutionScale = 20; % The demagnification scale of tubelens and objective. f_tube/f_objective
lambda = 1e-6;  % Wavelength
focal_SLM = 0.2; % focal length of the lens after slm.
psSLM = 20e-6;      % Pixel Size (resolution) at the scattered 3D region
Nx = 600;       % Number of pixels in X direction
Ny = 800;       % Number of pixels in Y direction


psXHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Nx;      % Pixel Size (resolution) at the scattered 3D region
psYHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Ny;      % Pixel Size (resolution) at the scattered 3D region

useGPU = 1;     % Use GPU to accelerate computation. Effective when Nx, Ny is large (e.g. 600*800).


z = [400 :4: 600] * 1e-6 ;   % Depth level requested in 3D region.
nfocus = 20;                % z(nfocus) denotes the depth of the focal plane.
thresholdh = 20000000;          % Intensity required to activate neuron.
thresholdl = 0;             % Intensity required to not to activate neuron.

%% Generate GIF
filename = 'source_phase_result_slmFocal_20_600by800_A_randomsource';
load(filename)
Ividmeas = zeros(Nx, Ny, numel(z));
usenoGPU = 0;
high = 3e4;

source = ones(Nx, Ny)/sqrt(Nx*Ny);

%figure();
for i = 1:numel(z)
    HStack = GenerateFresnelPropagationStack(Nx,Ny,z(i) - z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
    imagez = fresnelProp(phase2, source2, HStack);
    Ividmeas(:,:,i) = imagez;
%    imagesc(imagez);colormap gray;title(sprintf('Distance z %d', z(i)));
    colorbar;
    caxis([0, high]);
%     filename = sprintf('pointTarget%d.png', i);
%     print(['data/' filename], '-dpng')
%    pause(0.1);
end
figure();imagesc(source2);colorbar
Ividmeas(Ividmeas > high) = high;
Ividmeas = floor(Ividmeas/high * 63);


% hlimit = max(max(max(Ividmeas)));
% llimit = min(min(min(Ividmeas)));
% Ividmeas = floor((Ividmeas - llimit)/(hlimit - llimit) * 130);
% Ividmeas(Ividmeas > 63) = 63;
map = colormap(gray);

gifname = ['GS/gif/' filename  '.gif'];
for i = 1:numel(z)
    
    if i == 1;
        imwrite(Ividmeas(:,:,i), map, gifname, 'LoopCount',Inf, 'DelayTime', 0.1);
    else
        imwrite(Ividmeas(:,:,i), map, gifname, 'WriteMode','append','DelayTime',0.1);
    end
end

