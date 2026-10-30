function evaluate_batch(job_file)
% Evaluate explicit cases; absent annotations produce missing metrics.
job = jsondecode(fileread(job_file));
parameters = job.parameters.evaluation;
count = numel(job.cases);
records = cell(count,13);
for index = 1:count
    entry = job.cases(index);
    reference = im2double(imread(entry.reference));
    result = im2double(imread(entry.result));
    whole = image_metrics(reference,result,parameters);
    continuous=[]; rois=[];
    if ~isempty(entry.scale_map), continuous=readmatrix(entry.scale_map); end
    if ~isempty(entry.rois), rois=readmatrix(entry.rois); end
    regional = region_metrics(result,continuous,rois,parameters);
    partition = struct('roi_mean',NaN,'background_mean',NaN,'background_roi_ratio',NaN);
    if ~isempty(entry.tu) && ~isempty(entry.mask)
        mask = imread(entry.mask)>0;
        assert(isequal(size(mask),size(result)), 'Partition ROI mask must match the image');
        partition = partition_metrics(readmatrix(entry.tu),mask);
    end
    records(index,:) = {string(entry.sample_id),string(entry.group),whole.global_cnr,whole.enl,whole.epi,whole.fsim, ...
        regional.cnr,regional.fwhm_x,regional.fwhm_y,partition.roi_mean,partition.background_mean,partition.background_roi_ratio,NaN};
end
table = cell2table(records,'VariableNames',{'sample_id','group','global_cnr','enl','epi','fsim', ...
    'cnr','fwhm_x','fwhm_y','roi_tu_mean','background_tu_mean','tu_ratio','lpips'});
writetable(table,fullfile(job.output,'metrics.csv'));
summary = groupsummary(table,'group','mean',table.Properties.VariableNames(3:end));
summary.Properties.VariableNames = [{'group','count'},table.Properties.VariableNames(3:end)];
writetable(summary,fullfile(job.output,'group_means.csv'));
end
