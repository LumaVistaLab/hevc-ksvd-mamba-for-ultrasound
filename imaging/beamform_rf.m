function beamform_rf(job_file)
% Delay-and-sum beamforming with explicit physical acquisition metadata.
job = jsondecode(fileread(job_file));
parameters = job.parameters.imaging.rf;
files = dir(fullfile(job.input,'*.mat'));
assert(~isempty(files), 'No RF acquisitions');
for index = 1:numel(files)
    loaded = load(fullfile(files(index).folder,files(index).name),'acquisition');
    acquisition = loaded.acquisition;
    channels = double(acquisition.channels);
    frequency = acquisition.center_frequency_hz;
    sampling = acquisition.sampling_frequency_hz;
    count = acquisition.samples_per_line;
    elements = parameters.elements;
    assert(size(channels,1)==count*job.beams && size(channels,2)==elements, 'RF channel dimensions disagree');
    assert(mod(elements,2)==0 && frequency>0 && sampling>0, 'Invalid acquisition parameters');
    speed = parameters.sound_speed;
    aperture = parameters.aperture_cycles*speed/frequency;
    pitch = aperture/(elements-1);
    spacing = aperture/job.beams;
    pixel_depth = speed/sampling/2;
    depth = (0:count-1)'*pixel_depth;
    element_x = (-(elements-1)/2:(elements-1)/2)*pitch;
    offsets = (0:elements-1)*count;
    window = zeros(1,elements*2);
    window(1:elements) = hanning(elements)';
    radiofrequency = zeros(count,job.beams);
    for beam = 1:job.beams
        shift = floor((beam-1)*spacing/pitch)+1;
        rolled = circshift(window,shift);
        apodization = rolled(elements/2:elements*3/2-1);
        samples = channels((beam-1)*count+(1:count),1:elements).*apodization;
        position = -aperture/2+spacing/2+(beam-1)*spacing;
        distance = abs(position-element_x);
        delay = (acquisition.transmit_delay_cycles+parameters.delay_cycle_offset)/frequency;
        pointers = round((depth/speed+sqrt(distance.^2+depth.^2)/speed+delay)*sampling);
        pointers = min(count,max(1,pointers));
        radiofrequency(:,beam) = sum(samples(pointers+offsets),2);
    end
    radiofrequency = radiofrequency/max(max(abs(radiofrequency(:))),eps);
    envelope = abs(hilbert(radiofrequency));
    maximum = max(envelope(:));
    if maximum==0
        image = zeros(parameters.output_height,max(2,floor(aperture/(count*pixel_depth)*parameters.output_height)));
    else
        compressed = parameters.log_scale*(20*log10(max(envelope/maximum,realmin))+parameters.dynamic_range_db);
        height = parameters.output_height;
        width = floor(aperture/(count*pixel_depth)*height);
        image = zeros(height,width);
        for column = 1:width
            x = (column-1)*aperture/(width-1)/spacing+1;
            for row = 1:height
                z = (row-0.5)*(count*pixel_depth)/height/pixel_depth;
                x0 = floor(x); z0 = floor(z);
                if x0>=1 && x0+1<=job.beams && z0>=1 && z0+1<=count
                    fx=x-x0; fz=z-z0;
                    image(row,column) = (1-fx)*((1-fz)*compressed(z0,x0)+fz*compressed(z0+1,x0)) ...
                        +fx*((1-fz)*compressed(z0,x0+1)+fz*compressed(z0+1,x0+1));
                end
            end
        end
        image = image/max(max(image(:)),eps);
    end
    [~,identifier] = fileparts(files(index).name);
    assert(~isempty(regexp(identifier,'^sample_[0-9]{6}$','once')), 'Use a neutral RF identifier');
    imwrite(image,fullfile(job.output,[identifier '.tif']));
end
end
