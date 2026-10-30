function result = reconstruct_image(image, training_image, scales, parameters, gaussian_fusion)
% Train one dictionary per active scale and blend overlapping reconstructions.
[height,width] = size(image);
sizes = unique(scales(:))';
assert(all(mod(sizes,2)==0) && all(sizes>0), 'Patch sizes must be positive and even');
padding = max(sizes)/2;
training = padarray(training_image,[padding padding],'symmetric');
original = padarray(image,[padding padding],'symmetric');
dictionaries = cell(size(sizes)); windows = cell(size(sizes)); locations = cell(size(sizes));
for index = 1:numel(sizes)
    size_px = sizes(index);
    [rows,columns] = find(scales==size_px);
    if ~gaussian_fusion
        [~,order] = sortrows([rows,columns],[1 2]);
        rows = rows(order); columns = columns(order);
    end
    locations{index} = [rows,columns];
    patches = zeros(size_px^2,numel(rows));
    for patch = 1:numel(rows)
        r = rows(patch)+padding-size_px/2+1;
        c = columns(patch)+padding-size_px/2+1;
        patches(:,patch) = reshape(training(r:r+size_px-1,c:c+size_px-1),[],1);
    end
    dictionaries{index} = dictionary_learning(patches,parameters.dictionary);
    window = ones(size_px);
    if gaussian_fusion
        [x,y] = meshgrid(1:size_px,1:size_px);
        center = (size_px+1)/2;
        sigma = parameters.fusion_sigma_ratio*size_px;
        window = exp(-((x-center).^2+(y-center).^2)/(2*sigma^2));
    end
    windows{index} = window;
end
accumulated = zeros(height,width); weights = zeros(height,width);
limit = round(parameters.dictionary.atoms*parameters.dictionary.sparsity_ratio);
for index = 1:numel(sizes)
    size_px = sizes(index); dictionary = dictionaries{index};
    for patch = 1:size(locations{index},1)
        r = locations{index}(patch,1); c = locations{index}(patch,2);
        rp = r+padding-size_px/2+1; cp = c+padding-size_px/2+1;
        vector = reshape(original(rp:rp+size_px-1,cp:cp+size_px-1),[],1);
        coefficients = sparse_encode(dictionary,vector,limit,parameters.dictionary.residual_tolerance);
        restored = reshape(dictionary*coefficients,size_px,size_px);
        first_row = r-size_px/2+1; first_column = c-size_px/2+1;
        rows = max(1,first_row):min(height,first_row+size_px-1);
        columns = max(1,first_column):min(width,first_column+size_px-1);
        local_rows = rows-first_row+1; local_columns = columns-first_column+1;
        window = windows{index}(local_rows,local_columns);
        accumulated(rows,columns) = accumulated(rows,columns)+restored(local_rows,local_columns).*window;
        weights(rows,columns) = weights(rows,columns)+window;
    end
end
assert(all(weights(:)>0), 'Uncovered reconstruction pixels');
result = accumulated./weights;
end
