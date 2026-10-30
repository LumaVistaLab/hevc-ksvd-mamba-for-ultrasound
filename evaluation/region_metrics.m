function metrics = region_metrics(image, continuous, rois, parameters)
% Scale-conditioned CNR and drift-corrected point-response widths.
image = double(image);
metrics.cnr = NaN; metrics.fwhm_x = NaN; metrics.fwhm_y = NaN;
if ~isempty(continuous)
    assert(isequal(size(image),size(continuous)) && all(isfinite(continuous(:))), 'Invalid scale map');
    target = continuous>=parameters.cnr_target(1) & continuous<parameters.cnr_target(2);
    background = continuous>parameters.cnr_background(1) & continuous<=parameters.cnr_background(2);
    noise = continuous>=parameters.cnr_noise(1) & continuous<=parameters.cnr_noise(2);
    if any(target(:)) && any(background(:)) && any(noise(:))
        metrics.cnr = abs(mean(image(target))-mean(image(background)))/max(std(image(noise)),eps);
    end
end
widths = nan(size(rois,1),2);
for index = 1:size(rois,1)
    roi = centered_roi(image,rois(index,:),parameters);
    widths(index,1) = axis_width(image,roi,1,parameters);
    widths(index,2) = axis_width(image,roi,2,parameters);
end
if ~isempty(widths)
    metrics.fwhm_x = mean(widths(:,1),'omitnan');
    metrics.fwhm_y = mean(widths(:,2),'omitnan');
end
metrics.roi_widths = widths;
end

function roi = centered_roi(image,roi,parameters)
[height,width] = size(image);
roi = round(roi);
assert(numel(roi)==4 && all(isfinite(roi)) && all(roi(3:4)>0) && roi(3)<=width && roi(4)<=height, 'Invalid ROI');
roi(1) = max(0,min(width-roi(3),roi(1)));
roi(2) = max(0,min(height-roi(4),roi(2)));
region = image(roi(2)+(1:roi(4)),roi(1)+(1:roi(3)));
bright = region>=parameters.roi_brightness_fraction*max(region(:));
[rows,columns] = find(bright);
weights = region(bright);
if sum(weights)>eps
    displacement = round([sum(columns.*weights),sum(rows.*weights)]/sum(weights)-(roi(3:4)+1)/2);
    limit = floor(roi(3:4)*parameters.roi_drift_fraction);
    displacement = max(-limit,min(limit,displacement));
    roi(1:2) = max([0 0],min([width height]-roi(3:4),roi(1:2)+displacement));
end
end

function width_px = axis_width(image,roi,axis,parameters)
center_row = max(1,min(size(image,1),roi(2)+round((roi(4)+1)/2)));
center_column = max(1,min(size(image,2),roi(1)+round((roi(3)+1)/2)));
if axis==1
    signal = image(center_row,:); left=roi(1)+1; right=roi(1)+roi(3); span=roi(3);
else
    signal = image(:,center_column)'; left=roi(2)+1; right=roi(2)+roi(4); span=roi(4);
end
step = max(1,ceil(span*parameters.roi_expansion_fraction));
while true
    profile = signal(left:right);
    threshold = max(profile)*parameters.half_height_fraction;
    selected = find(profile>=threshold);
    if isempty(selected), width_px=NaN; return; end
    first=selected(1); last=selected(end);
    bounded = first>1 && last<numel(profile);
    if bounded || (left==1 && right==numel(signal)), break; end
    left=max(1,left-step); right=min(numel(signal),right+step);
end
lower = first; upper = last;
if first>1
    difference = profile(first)-profile(first-1);
    if abs(difference)>=eps, lower=first-1+(threshold-profile(first-1))/difference; end
end
if last<numel(profile)
    difference = profile(last+1)-profile(last);
    if abs(difference)>=eps, upper=last+(threshold-profile(last))/difference; end
end
width_px = upper-lower;
end
