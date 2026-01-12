addpath(genpath('~/Documents/MATLAB/spm12/'))

blocks = 1:30;  % Use colon notation instead of linspace
Y_raw = [];
Y_filt = [];

% Set voxel coordinates (in voxel space)
x = 63; y = 14; z = 29;

voxel_id = [];  % to be set after loading the first block

for block = blocks
    fprintf('doing block...%d\n', block)

    nii_file = sprintf('/cifs/diedrichsen/data/Chord_exp/EFC_learningfMRI/imaging_data/subj105/usubj105_run_%02d.nii', block);
    V = spm_vol(nii_file);
    Y = spm_read_vols(V);
    [nx, ny, nz, nt] = size(Y);  % nt = number of timepoints

    if isempty(voxel_id)
        voxel_id = sub2ind([nx, ny, nz], x, y, z);  % compute once
    end

    Y2D = reshape(Y, [], nt)';  % now nt x nvox

    Y_raw = [Y_raw; Y2D(:, voxel_id)];

    % Create high-pass filter (128s cutoff, 1s TR)
    K = spm_filter(struct(...
        'RT', 1, ...
        'row', 1:nt, ...
        'HParam', 128));

    Y_filt_tmp = spm_filter(K, Y2D);
    Y_filt = [Y_filt; Y_filt_tmp(:, voxel_id)];

    if any(isnan(Y_filt_tmp(:, voxel_id))) || any(isinf(Y_filt_tmp(:, voxel_id)))
        warning('Filtered data contains NaNs or Infs in block %d', block);
    end
end

% Plot raw and filtered timecourse
figure;
plot(Y_raw, 'k'); hold on
plot(Y_filt, 'r')
legend('Raw', 'Filtered')
title(sprintf('Timecourse at voxel (%d, %d, %d)', x, y, z))
xlabel('Timepoints')
ylabel('Signal intensity')
