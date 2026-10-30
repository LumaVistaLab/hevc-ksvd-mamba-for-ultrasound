function batch_denoise(job_file)
% Process a neutral image tree using a JSON job supplied by the launcher.
job = jsondecode(fileread(job_file));
settings = job.parameters.block_ksvd;
sizes = reshape(settings.profiles.(job.profile),1,[]);
assert(all(diff(sizes)>0) && all(mod(sizes,2)==0), 'Invalid configured scales');
rng(job.parameters.seed,'twister');
files = [dir(fullfile(job.images,'**','*.tif'));dir(fullfile(job.images,'**','*.tiff'))];
assert(~isempty(files), 'No input TIFF images');
records = cell(numel(files),3);
for index = 1:numel(files)
    source = fullfile(files(index).folder,files(index).name);
    relative = erase(source,[char(job.images) filesep]);
    [folder,identifier] = fileparts(relative);
    image = im2double(imread(source));
    assert(ismatrix(image), 'Expected grayscale input');
    start = tic;
    noise = settings.noise;
    draw = rand(size(image));
    pepper = imdilate(draw<noise.impulse_density/2,strel('square',noise.dilation_size));
    salt = imdilate(draw>=noise.impulse_density/2 & draw<noise.impulse_density,strel('square',noise.dilation_size));
    training = image; training(pepper)=0; training(salt)=1;
    training = training+noise.gaussian_sigma*randn(size(image));
    if job.fixed
        for size_px = sizes
            restored = reconstruct_image(image,training,size_px*ones(size(image)),settings,false);
            save_image(restored,fullfile(job.output,'images',sprintf('fixed_%02d',size_px),folder,[identifier '.tif']));
        end
    else
        tu = readmatrix(fullfile(job.priors,folder,[identifier '_tu.csv']));
        pu = readmatrix(fullfile(job.priors,folder,[identifier '_pu.csv']));
        [continuous,discrete] = scale_map(image,tu,pu,sizes,settings.scale);
        restored = reconstruct_image(image,training,discrete,settings,true);
        save_image(restored,fullfile(job.output,'images',folder,[identifier '.tif']));
        map_root = fullfile(job.output,'scale_maps',folder);
        if ~isfolder(map_root), mkdir(map_root); end
        writematrix(continuous,fullfile(map_root,[identifier '_continuous.csv']));
        writematrix(discrete,fullfile(map_root,[identifier '_discrete.csv']));
    end
    records(index,:) = {string(relative),job.profile,toc(start)};
    fprintf('Processed %d/%d\n',index,numel(files));
end
writetable(cell2table(records,'VariableNames',{'image','profile','seconds'}),fullfile(job.output,'timing.csv'));
end

function save_image(image,path)
folder = fileparts(path);
if ~isfolder(folder), mkdir(folder); end
imwrite(image,path);
end
