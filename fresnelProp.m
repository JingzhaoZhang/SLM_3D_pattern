  function [ imagez ] = fresnelProp( phase, source, Hstack)
%FRESNELPROP generate fresnel propagation of light with PHASE and constant
%Inteisty. HStack needs to have low frequency component at the center.
    objectField = exp(1i * phase);
    imagez = abs(fftshift(fft2(objectField .* Hstack)).^2);
    imagez = ifft2(fft2(source) .* fft2(ifftshift(imagez)));
    %imagez = ifft2(fft2(ifftshift(imagez)));
end
 
