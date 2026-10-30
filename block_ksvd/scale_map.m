function [continuous, discrete] = scale_map(image, tu, pu, sizes, parameters)
% Interpolate TU centers, adjust dark PU regions, smooth and select scales.
[height,width] = size(image);
validate_blocks(tu,height,width);
validate_blocks(pu,height,width);
centers = tu(:,2:3) + tu(:,1)/2;
keys = round(centers(:,2))*(width+1) + round(centers(:,1));
[~,~,group] = unique(keys);
points = zeros(max(group),2);
values = inf(max(group),1);
for index = 1:numel(group)
    if tu(index,1)<values(group(index))
        points(group(index),:) = centers(index,:);
        values(group(index)) = tu(index,1);
    end
end
[x,y] = meshgrid(0:width-1,0:height-1);
query = [x(:),y(:)];
if size(points,1)<3 || rank(points-points(1,:))<2
    nearest = nearest_points(points,query);
    interpolated = values(nearest);
else
    triangulation = delaunayTriangulation(points);
    [triangle,barycentric] = pointLocation(triangulation,query);
    valid = ~isnan(triangle);
    interpolated = zeros(size(query,1),1);
    vertices = triangulation.ConnectivityList(triangle(valid),:);
    interpolated(valid) = sum(barycentric(valid,:).*values(vertices),2);
    interpolated(~valid) = values(dsearchn(points,query(~valid,:)));
end
interpolated = reshape(interpolated,height,width);
if parameters.zoom_strength>0
    gray = zeros(size(pu,1),1);
    membership = zeros(height,width);
    for index = 1:size(pu,1)
        columns = pu(index,2)+1:min(width,pu(index,2)+pu(index,1));
        rows = pu(index,3)+1:min(height,pu(index,3)+pu(index,1));
        region = image(rows,columns);
        gray(index) = mean(region(:));
        membership(rows,columns) = index;
    end
    missing = membership==0;
    if any(missing(:))
        centers = pu(:,2:3)+pu(:,1)/2;
        membership(missing) = nearest_points(centers,[x(missing),y(missing)]);
    end
    brightness = reshape(gray(membership(:)),height,width)/max(max(gray),eps);
    zoom = 1+parameters.zoom_strength*(1-brightness).^parameters.zoom_gamma ...
        .*(parameters.structure_limit./interpolated-1);
    interpolated = interpolated.*zoom;
end
if parameters.smoothing_radius>0
    kernel = fspecial('average',2*parameters.smoothing_radius+1);
    interpolated = imfilter(interpolated,kernel,'symmetric');
end
lower = min(interpolated(:)); upper = max(interpolated(:));
if upper-lower>parameters.constant_tolerance
    continuous = parameters.lower_bound+(interpolated-lower)/(upper-lower) ...
        *(max(sizes)-parameters.lower_bound);
else
    continuous = min(sizes)*ones(height,width);
end
[~,labels] = min(abs(continuous(:)-reshape(sizes,1,[])),[],2);
discrete = reshape(sizes(labels),height,width);
end

function nearest = nearest_points(points,query)
nearest = ones(size(query,1),1); distance = inf(size(query,1),1);
for index = 1:size(points,1)
    candidate = sum((query-points(index,:)).^2,2);
    better = candidate<distance;
    nearest(better) = index; distance(better) = candidate(better);
end
end

function validate_blocks(blocks,height,width)
assert(size(blocks,2)==3 && ~isempty(blocks) && all(isfinite(blocks(:))), 'Invalid partition array');
assert(all(blocks(:)==floor(blocks(:))) && all(blocks(:,1)>0), 'Partition values must be integral');
assert(all(blocks(:,2)>=0 & blocks(:,2)<width) && all(blocks(:,3)>=0 & blocks(:,3)<height), 'Partition outside image');
end
