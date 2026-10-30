function metrics = partition_metrics(blocks, mask)
% Classify each TU by its center pixel and summarize block sizes.
[height,width] = size(mask);
assert(size(blocks,2)==3 && all(isfinite(blocks(:))) && all(blocks(:)==floor(blocks(:))), 'Invalid TU array');
columns = min(width,max(1,blocks(:,2)+floor(blocks(:,1)/2)+1));
rows = min(height,max(1,blocks(:,3)+floor(blocks(:,1)/2)+1));
inside = logical(mask(sub2ind([height width],rows,columns)));
metrics.roi_mean = 0; metrics.background_mean = 0;
if any(inside), metrics.roi_mean=mean(blocks(inside,1)); end
if any(~inside), metrics.background_mean=mean(blocks(~inside,1)); end
metrics.background_roi_ratio = Inf;
if metrics.roi_mean>0, metrics.background_roi_ratio=metrics.background_mean/metrics.roi_mean; end
end
