function [dictionary, coefficients, errors] = dictionary_learning(patches, parameters)
% Alternate sparse coding and rank-one dictionary updates.
atoms = parameters.atoms;
limit = round(atoms*parameters.sparsity_ratio);
assert(atoms > 0 && parameters.iterations > 0, 'Invalid dictionary dimensions');
dictionary = randn(size(patches,1),atoms);
dictionary = dictionary - min(dictionary(:));
dictionary = dictionary ./ max(sqrt(sum(dictionary.^2,1)),eps);
coefficients = sparse_encode(dictionary,patches,limit,parameters.residual_tolerance);
errors = zeros(parameters.iterations,1);
previous = 0;
for iteration = 1:parameters.iterations
    residual = patches - dictionary*coefficients;
    for atom = 1:atoms
        active = find(coefficients(atom,:)~=0);
        if isempty(active), active = 1:size(patches,2); end
        local = residual(:,active) + dictionary(:,atom)*coefficients(atom,active);
        [direction,strength,weights] = svds(local,1);
        dictionary(:,atom) = direction;
        coefficients(atom,active) = strength*weights';
        residual(:,active) = local - dictionary(:,atom)*coefficients(atom,active);
    end
    coefficients = sparse_encode(dictionary,patches,limit,parameters.residual_tolerance);
    difference = patches - dictionary*coefficients;
    errors(iteration) = sqrt(sum(sum(difference.^2))/numel(patches));
    if abs(errors(iteration)-previous)<parameters.convergence_tolerance, break; end
    previous = errors(iteration);
end
errors = errors(1:iteration);
end
