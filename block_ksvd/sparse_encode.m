function coefficients = sparse_encode(dictionary, patches, limit, tolerance)
% Greedy correlation selection with the prescribed residual stopping rule.
assert(limit >= 1 && limit <= size(dictionary,2), 'Invalid sparsity limit');
coefficients = zeros(size(dictionary,2),size(patches,2));
for sample = 1:size(patches,2)
    residual = patches(:,sample);
    selected = false(size(dictionary,2),1);
    for step = 1:limit
        correlation = dictionary' * residual;
        score = abs(correlation);
        score(selected) = -Inf;
        [~,atom] = max(score);
        coefficients(atom,sample) = correlation(atom);
        selected(atom) = true;
        residual = residual - correlation(atom)*dictionary(:,atom);
        if sqrt(residual'*residual) <= tolerance, break; end
    end
end
end
