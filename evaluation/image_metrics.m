function metrics = image_metrics(reference, result, parameters)
% Whole-image statistics and structure similarity on aligned grayscale arrays.
assert(isequal(size(reference),size(result)) && ismatrix(result), 'Images must be aligned grayscale');
reference = double(reference); result = double(result);
pixels = result(:);
metrics.global_cnr = mean(pixels)/max(std(pixels),eps);
metrics.enl = mean(pixels)^2/max(var(pixels),eps);
sobel = [-1 0 1;-2 0 2;-1 0 1];
first = gradient_magnitude(reference,sobel);
second = gradient_magnitude(result,sobel);
if std(first(:))<eps || std(second(:))<eps
    metrics.epi = 1;
else
    correlation = corrcoef(first(:),second(:));
    metrics.epi = correlation(1,2);
end
scharr = [3 0 -3;10 0 -10;3 0 -3]/16;
first_gradient = gradient_magnitude(reference,scharr);
second_gradient = gradient_magnitude(result,scharr);
first_phase = phase_response(reference,parameters);
second_phase = phase_response(result,parameters);
gradient_similarity = (2*first_gradient.*second_gradient+parameters.fsim_gradient_constant) ...
    ./(first_gradient.^2+second_gradient.^2+parameters.fsim_gradient_constant);
phase_similarity = (2*first_phase.*second_phase+parameters.fsim_phase_constant) ...
    ./(first_phase.^2+second_phase.^2+parameters.fsim_phase_constant);
importance = max(first_phase,second_phase);
metrics.fsim = sum(gradient_similarity(:).*phase_similarity(:).*importance(:))/max(sum(importance(:)),eps);
end

function magnitude = gradient_magnitude(image,kernel)
magnitude = hypot(imfilter(image,kernel,'symmetric'),imfilter(image,kernel','symmetric'));
end

function response = phase_response(image,parameters)
response = zeros(size(image));
for sigma = reshape(parameters.fsim_scales,1,[])
    size_px = 2*ceil(parameters.fsim_kernel_radius*sigma)+1;
    response = response+abs(imfilter(image,fspecial('log',size_px,sigma),'symmetric'));
end
response = response/numel(parameters.fsim_scales);
end
