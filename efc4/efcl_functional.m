function varargout = efcl_functional(what, varargin)
    
    
    
    % Use a different baseDir when using your local machine or the cbs
    % server. Add more directory if needed. Use single quotes ' and not
    % double quotes " because some spm function raise error with double
    % quotes
    if isfolder('/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/')
        baseDir = '/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/';
        
        addpath(genpath('~/Documents/GitHub/dataframe/'))
        addpath(genpath('~/Documents/GitHub/spmj_tools/'))
        addpath(genpath('~/Documents/MATLAB/spm12/'))
        
    elseif isfolder('/path/to/project/cifs/directory/')
        baseDir = '/path/to/project/cifs/directory/';
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end
    
    sn = [];
    day = [];
    rtm = 0;
    prefix = 'u';
    vararginoptions(varargin,{'sn', 'day', 'rtm', 'prefix'})
    
    anatomicalDir = 'anatomicals';
    bidsDir = 'BIDS'; % Raw data post AutoBids conversion
    imagingRawDir = 'imaging_data_raw'; % Temporary directory for raw functional data
    imagingDir = 'imaging_data'; % Preprocesses functional data
    fmapDir = 'fieldmaps'; % Fieldmap dir after moving from BIDS and SPM make fieldmap

    pinfo = dload(fullfile(baseDir,'participants.tsv'));

    % get participant row from participant.tsv
    subj_row=getrow(pinfo, pinfo.sn== sn);
    
    % get subj_id
    subj_id = subj_row.participant_id{1};
    
    % get day_id
    day_id = sprintf('day%d', day);

    % get runs (FuncRuns column needs to be in participants.tsv)    
    runs = spmj_dotstr2array(subj_row.FuncRuns{1});

    switch(what)
        case 'BIDS:move_unzip_raw_func'
            % Moves, unzips and renames raw functional (BOLD) images from 
            % BIDS directory. After you run this function you will find
            % nRuns Nifti files named <subj_id>_run_XX.nii in the 
            % <project_id>/imaging_data_raw/<subj_id>/ directory.
            
            % handling input args:
            if isempty(sn)
                error('BIDS:move_unzip_raw_func -> ''sn'' must be passed to this function.')
            end
            
            if isempty(day)
                error('BIDS:move_unzip_raw_func -> ''day'' must be passed to this function.')
            end

            % loop on runs of sess:
            for run = runs
                
                % pull functional raw name from the participant.tsv:
                FuncRawName_tmp = [pinfo.FuncRawName{pinfo.sn==sn} '.nii.gz'];  

                % add run number to the name of the file:
                FuncRawName_tmp = replace(FuncRawName_tmp,'XX',sprintf('%.02d',run));

                % path to the subj func data:
                func_raw_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn), day_id, 'func',FuncRawName_tmp);
        
                % destination path:
                output_folder = fullfile(baseDir,imagingRawDir,subj_id, day_id);
                output_file = fullfile(output_folder,[subj_id sprintf('_run_%.02d.nii.gz',run)]);
                
                if ~exist(output_folder,"dir")
                    mkdir(output_folder);
                end
                
                % copy file to destination:
                [status,msg] = copyfile(func_raw_path,output_file);
                if ~status  
                    error('FUNC:move_unzip_raw_func -> subj %d raw functional (BOLD) was not moved from BIDS to the destenation:\n%s',sn,msg)
                end
        
                % unzip the .gz files to make usable for SPM:
                gunzip(output_file);
        
                % delete the compressed file:
                delete(output_file);
            end   
    
        case 'BIDS:move_unzip_raw_fmap'
            % Moves, unzips and renames raw fmap images from BIDS
            % directory. After you run this function you will find two
            % files named <subj_id>_phase.nii and <subj_id>_magnitude.nii
            % in the <project_id>/fieldmaps/<subj_id>/ directory. The
            % <subj_id>_phase.nii contains phase information derived from
            % the MRI signal, reflecting local magnetic field
            % inhomogeneities caused by factors such as tissue
            % susceptibility differences (e.g., at air-tissue interfaces in
            % the nasal cavities or sinuses). This phase information can be
            % used to compute a fieldmap, which is essential for correcting
            % geometric distortions (unwarping) in other MRI sequences.
            
            if isempty(sn)
                error('BIDS:move_unzip_raw_fmap -> ''sn'' must be passed to this function.')
            end
            
            if isempty(day)
                error('BIDS:move_unzip_raw_func -> ''day'' must be passed to this function.')
            end
            
            % pull fmap raw names from the participant.tsv:
            fmapMagnitudeName_tmp = pinfo.fmapMagnitudeName{pinfo.sn==sn};
            magnitude = [fmapMagnitudeName_tmp '.nii.gz'];
            
            fmapPhaseName_tmp = pinfo.fmapPhaseName{pinfo.sn==sn};
            phase = [fmapPhaseName_tmp '.nii.gz'];

            % path to the subj fmap data:
            magnitude_path = fullfile(baseDir,bidsDir, sprintf('subj%.02d',sn),day_id,'fmap',magnitude);
            phase_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn),day_id,'fmap',phase);
    
            % destination path:
            output_folder = fullfile(baseDir,fmapDir,subj_id, day_id);
            output_magnitude = fullfile(output_folder,[subj_id '_magnitude.nii.gz']);
            output_phase = fullfile(output_folder,[subj_id '_phase.nii.gz']);
            
            if ~exist(output_folder,"dir")
                mkdir(output_folder);
            end
            
            % copy magnitude to destination:
            [status,msg] = copyfile(magnitude_path,output_magnitude);
            if ~status  
                error('BIDS:move_unzip_raw_fmap -> subj %d, fmap magnitude was not moved from BIDS to the destenation:\n%s',sn,msg)
            end
            % unzip the .gz files to make usable for SPM:
            gunzip(output_magnitude);
    
            % delete the compressed file:
            delete(output_magnitude);

            % copy phase to destination:
            [status,msg] = copyfile(phase_path,output_phase);
            if ~status  
                error('BIDS:move_unzip_raw_fmap -> subj %d, fmap phase was not moved from BIDS to the destenation:\n%s',sn,msg)
            end
            % unzip the .gz files to make usable for SPM:
            gunzip(output_phase);
    
            % delete the compressed file:
            delete(output_phase);
    
    
        case 'FUNC:make_fmap' 
            % Differences in magnetic susceptibility between tissues (e.g.,
            % air-tissue or bone-tissue interfaces) can cause
            % inhomogeneities in the magnetic field. These inhomogeneities
            % result in spatial distortions along the phase-encoding
            % direction, which is the direction in which spatial location
            % is encoded using a phase gradient. To account for these
            % distortions, this step generates a Voxel Displacement Map
            % (VDM) for each run, saved as files named
            % vdm5_sc<subj_id>_phase_run_XX.nii in the fieldmap directory.
            % 
            % The VDM assigns a value in millimeters to each voxel,
            % indicating how far it should be shifted along the
            % phase-encoding direction to correct for the distortion. If
            % you open the VDM in FSLeyes, you will notice that the
            % distortion is particularly strong in the temporal lobe due to
            % proximity to the nasal cavities, where significant
            % differences in magnetic susceptibility occur.
            % 
            % In the fieldmap directory, you will also find the intermediate
            % files bmask<subj_id>_magnitude.nii and
            % fpm_sc<subj_id>_phase.nii that are used for VDM calculation
            % 
            % In the imaging_data_raw directory, you will find unwarped
            % functional volumes named u<subj_id>_run_XX.nii. These
            % correspond to the corrected first volume of each functional
            % run. Open them in FSL to inspect how the distortion was
            % corrected using the VDM (this step is for quality checking;
            % the actual unwarping is performed in a later step).
            % 
            % In addition, the imaging_raw_data directory contains the
            % intermediate file wfmag_<subj_id>_run_XX.nii that is
            % necessary to perform unwarping in eah run.
            
            if isempty(sn)
                error('FUNC:make_fmap -> ''sn'' must be passed to this function.')
            end
            
            if isempty(day)
                error('BIDS:move_unzip_raw_func -> ''day'' must be passed to this function.')
            end
            
            epi_files = {}; % Initialize as an empty cell array
            for run = runs
                epi_files{end+1} = sprintf('%s_run_%02d.nii', subj_id, run);
            end

            [et1, et2, tert] = spmj_et1_et2_tert(fullfile(baseDir, bidsDir, subj_id, day_id, 'fmap'),...
                                                fullfile(baseDir, bidsDir, subj_id, day_id, 'func'),...
                                                subj_row.dicom_id{1});

            spmj_makefieldmap(fullfile(baseDir, fmapDir, subj_id, day_id), ...
                              sprintf('%s_magnitude.nii', subj_id),...
                              sprintf('%s_phase.nii', subj_id),...
                              'phase_encode', -1, ... % It's -1 (A>>P) or 1 (P>>A) and can be found in imaging sequence specifications
                              'et1', et1, ...
                              'et2', et2, ...
                              'tert', tert, ...
                              'func_dir',fullfile(baseDir, imagingRawDir, subj_id, day_id),...
                              'epi_files', epi_files);
          
          % change name or (current version of) realign_unwarp wont work
          for run=1:numel(runs)
              oldName = sprintf('vdm5_sc%s_phase_run_%d.nii', subj_id, run);
              newName = sprintf('vdm5_sc%s_%s_phasediff_run_%d.nii', subj_id, day_id, run);
              movefile(fullfile(baseDir, fmapDir, subj_id, day_id, oldName), fullfile(baseDir, fmapDir, subj_id, day_id, newName))
          end
        
        case 'FUNC:realign_unwarp'      
            % Do spm_realign_unwarp
            
%             % handling input args:
%             sn = [];
%             rtm = 0;
%             vararginoptions(varargin,{'sn','rtm'})
            if isempty(sn)
                error('FUNC:make_fmap -> ''sn'' must be passed to this function.')
            end
            
            if isempty(day)
                error('BIDS:move_unzip_raw_func -> ''day'' must be passed to this function.')
            end

            run_list{1} = {}; % Initialize as an empty cell array 
            for run = runs
                run_list{1}{end+1} = sprintf('run_%02d', run);
            end

            spmj_realign_unwarp(subj_id, ...
                 run_list, ...
                'rawdata_dir',fullfile(baseDir,imagingRawDir),...
                'fmap_dir',fullfile(baseDir,fmapDir),...
                'raw_name','run',...
                'sess_names', {day_id},...
                'rtm',rtm);
        
    
        case 'FUNC:inspect_realign'
            % looks for motion correction logs into imaging_data, needs to
            % be run after realigned images are moved there from
            % imaging_data_raw

            % handling input args:
            sn = []; 
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('FUNC:inspect_realign_parameters -> ''sn'' must be passed to this function.')
            end
            
            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = ['rp_', subj_id, '_run_', run, '.txt'];
            end

            smpj_plot_mov_corr(run_list)

        case 'FUNC:move_realigned_images'          
            % Move images created by realign(+unwarp) into imaging_data
            
            % handling input args:
%             sn = [];
%             prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
%             rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
%             vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:move_realigned_images -> ''sn'' must be passed to this function.')
            end
            
            if isempty(day)
                error('FUNC:move_realigned_images -> ''day'' must be passed to this function.')
            end
            
            % loop on runs of the session:
            for run = runs
                % realigned (and unwarped) images names:
                file_name = sprintf('%s%s_run_%02d.nii',prefix, subj_id, run);
                source = fullfile(baseDir,imagingRawDir,subj_id,day_id,file_name);
                dest = fullfile(baseDir,imagingDir,subj_id, day_id);
                if ~exist(dest,'dir')
                    mkdir(dest)
                end

                file_name = file_name(length(prefix) + 1:end); % skip prefix in realigned (and unwarped) files
%                 dest = fullfile(baseDir,imagingDir,subj_id,file_name);
                % move to destination:
                [status,msg] = movefile(source,dest);
                if ~status  
                    error('FUNC:move_realigned_images -> %s',msg)
                end

                % realign parameters names:
                source = fullfile(baseDir,imagingRawDir,subj_id,day_id,sprintf('rp_%s_run_%02d.txt', subj_id,  run));
                dest = fullfile(baseDir,imagingDir,subj_id,day_id,sprintf('rp_%s_run_%02d.txt', subj_id,  run));
                % move to destination:
                [status,msg] = movefile(source,dest);
                if ~status  
                    error('FUNC:move_realigned_images -> %s',msg)
                end
            end
            
            % mean epi name - the generated file name will be different for
            % rtm=0 and rtm=1. Extra note: rtm is an option in
            % realign_unwarp function. Refer to spmj_realign_unwarp().
            if rtm==0   % if registered to first volume of each run:
                source = fullfile(baseDir,imagingRawDir,subj_id, day_id,sprintf('mean%s%s_run_02.nii', prefix, subj_id));
                dest = fullfile(baseDir,imagingDir,subj_id, day_id,sprintf('mean%s%s_run_02.nii', prefix, subj_id));
            else        % if registered to mean image of each run:
                source = fullfile(baseDir,imagingRawDir,subj_id,day_id,[prefix, 'meanepi_', subj_id, '.nii']);
                dest = fullfile(baseDir,imagingDir,subj_id,day_id,[prefix, 'meanepi_', subj_id, '.nii']);
            end
            % move to destination:
            [status,msg] = movefile(source,dest);
            if ~status  
                error('BIDS:move_realigned_images -> %s',msg)
            end
            % end
            
        case 'FUNC:meanimage_bias_correction'
            % EPI images often contain smooth artifacts caused by MRI
            % physics which make the intensity of signal from the same
            % tissue (e.g., grey matter, white matter) non-uniform. This
            % step perform bias correction and creates an image where the
            % signal from each tissue type is more uniform. This image is
            % then co-registered to the anatomical image. Bias correction
            % help make co-registration more accurate. If the realignment
            % was done with respect to the first volume of each run of each
            % session, the mean image will be calculated on the first run
            % of each session and will be called 'meanu*_run_01.nii'
            % ('mean' indicates the image is average of the volumes and 'u'
            % indicates it's unwarped). Therefore, we do the bias
            % correction on this file. But if you do the realignment to the
            % mean epi of every run, the generated mean file will be named
            % 'umeanepi_*' and we do the bias correction on this file. In
            % addition, this step generates five tissue probability maps
            % (c1-5) for grey matter, white matter, csf, bone and soft
            % tissue.
            
            % handling input args:
%             sn = [];
%             prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
%             rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
%             vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:meanimage_bias_correction -> ''sn'' must be passed to this function.')
            end

            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end

            if rtm==0   % if registered to first volume of each run:
                P{1} = fullfile(baseDir, imagingDir, subj_id,day_id, sprintf('mean%s%s_run_02.nii', prefix, subj_id));
            else        % if registered to mean image of each run:
                P{1} = fullfile(baseDir, imagingDir, subj_id,day_id, [prefix, 'meanepi_', subj_id, '.nii']);
            end
            spmj_bias_correct(P);
        
    
        case 'FUNC:coreg'                                                      
            % coregister rbumean image to anatomical image for each session

            % (1) Manually seed the functional/anatomical registration
            % - Open fsleyes
            % - Add anatomical image and b*mean*.nii (bias corrected mean) image to overlay
            % - click on the bias corrected mean image in the 'Overlay
            %   list' in the bottom left of the fsleyes window.
            %   list to highlight it.
            % - Open tools -> Nudge
            % - Manually adjust b*mean*.nii image to the anatomical by 
            %   changing the 6 paramters (tranlation xyz and rotation xyz) 
            %   and Do not change the scales! 
            % - When done, click apply and close the tool tab. Then to save
            %   the changes, click on the save icon next to the mean image 
            %   name in the 'Overlay list' and save the new image by adding
            %   'r' in the beginning of the name: rb*mean*.nii. If you don't
            %   set the format to be .nii, fsleyes automatically saves it as
            %   a .nii.gz so either set it or gunzip afterwards to make it
            %   compatible with SPM.
            
            % (2) Run automated co-registration to register bias-corrected meanimage to anatomical image
            
%             % handling input args:
%             sn = [];
%             prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
%             rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
%             vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:coreg -> ''sn'' must be passed to this function.')
            end

            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end

            if rtm==0   % if registered to first volume
                mean_file_name = sprintf('bmean%s%s_%s.nii', prefix, subj_id, run_list{1});
            else    % if registered to the mean image
                mean_file_name = sprintf('rb%smeanepi_%s.nii', prefix, subj_id);
            end
            J.source = {fullfile(baseDir,imagingDir,subj_id,day_id, mean_file_name)}; 
            J.ref = {fullfile(baseDir,anatomicalDir,subj_id,[subj_id, '_T1w','.nii'])};
            J.other = {''};
            J.eoptions.cost_fun = 'nmi';
            J.eoptions.sep = [4 2];
            J.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
            J.eoptions.fwhm = [7 7];
            matlabbatch{1}.spm.spatial.coreg.estimate=J;
            spm_jobman('run',matlabbatch);
                
        case 'FUNC:make_samealign'
            % align to registered bias corrected mean image of each session
            % (rb*mean*.nii). Alignment happens only by changing the
            % transform matrix in the header files of the functional 4D
            % .nii files to the transform matrix that aligns them to
            % anatomical. The reason that it works is: 1) in the
            % realignment (+unwarping) process, we have registered every
            % single volume of every single run to the first volume of the
            % first run of the session. 2) In the same step, for each
            % session, a mean functional image (meanepi*.nii or meanu*.nii
            % based on the rtm option) was generated. This mean image is
            % alread in the space of all the functional volumes. Later we
            % coregister this image to the anatomical space. Therefore, if
            % we change the transformation matrices of all the functional
            % volumes to the transform matrix of the coregistered image,
            % they will all tranform into the anatomical coordinates space.
    
%             % handling input args:
%             sn = [];
%             prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
%             rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
%             vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:make_samealign -> ''sn'' must be passed to this function.')
            end

            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end
            
            % select the reference image:
            if rtm==0
                P{1} = fullfile(baseDir,imagingDir,subj_id,day_id, sprintf('bmean%s%s_%s.nii', prefix, subj_id, run_list{1}));
            else
                P{1} = fullfile(baseDir,imagingDir,subj_id,day_id,['rb' prefix 'meanepi_' subj_id '.nii']);
            end

            % select images to be realigned:
            Q = {};
            for r = 1:length(run_list)
                for i = 1:pinfo.numTR
                     Q{end+1} = fullfile(baseDir,imagingDir,subj_id,day_id,sprintf('%s%s_%s.nii,%d', prefix, subj_id, run_list{r}, i));
                end
            end

            spmj_makesamealign_nifti(char(P),char(Q));
            % end
        
        case 'FUNC:make_maskImage'       
            % Make mask images (noskull and gray_only) for 1st level glm
            
%             % handling input args:
%             sn = [];
%             prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
%             rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
%             vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:make_maskImage -> ''sn'' must be passed to this function.')
            end

            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end
        
            % bias corrected mean epi image:
            if rtm==0
                nam{1} = fullfile(baseDir,imagingDir,subj_id,day_id,sprintf('bmean%s%s_%s.nii', prefix, subj_id, run_list{1}));
            else
                nam{1} = fullfile(baseDir,imagingDir,subj_id,day_id,['rb' prefix 'meanepi_' subj_id '.nii']);
            end
            nam{2}  = fullfile(baseDir,anatomicalDir,subj_id,['c1',subj_id, '_T1w','.nii']);
            nam{3}  = fullfile(baseDir,anatomicalDir,subj_id,['c2',subj_id, '_T1w','.nii']);
            nam{4}  = fullfile(baseDir,anatomicalDir,subj_id,['c3',subj_id, '_T1w','.nii']);
            spm_imcalc(nam, fullfile(baseDir,imagingDir,subj_id,day_id, 'rmask_noskull.nii'), 'i1>1 & (i2+i3+i4)>0.2')
            
%             source = fullfile(baseDir,imagingDir,subj_id, day_id,'rmask_noskull.nii'); % does this need to have some flag for session?
%             dest = fullfile(baseDir,anatomicalDir,subj_id,day_id,'rmask_noskull.nii');
%             movefile(source,dest);
            
            % gray matter mask for covariance estimation
            % ------------------------------------------
            nam={};
            % nam{1}  = fullfile(imagingDir,subj_id{sn}, 'sess1', ['rb' prefix 'meanepi_' subj_id{sn} '.nii']);

            % IS THIS CHANGE CORRECT??
            % nam{1}  = fullfile(baseDir,imagingDir,char(pinfo.subj_id(pinfo.sn==sn)),sprintf('sess%d',sess), ['rb' prefix 'meanepi_' char(pinfo.subj_id(pinfo.sn==sn)) '.nii']);
            % bias corrected mean epi image:
            if rtm==0
                nam{1} = fullfile(baseDir,imagingDir,subj_id,day_id,sprintf('bmean%s%s_%s.nii', prefix, subj_id, run_list{1}));
            else
                nam{1} = fullfile(baseDir,imagingDir,subj_id,day_id,['rb' prefix 'meanepi_' subj_id '.nii']);
            end

            nam{2}  = fullfile(baseDir,anatomicalDir,subj_id,['c1',subj_id, '_T1w','.nii']);
            spm_imcalc(nam, fullfile(baseDir,imagingDir,subj_id, day_id,'rmask_gray.nii'), 'i1>1 & i2>0.4')
            
%             source = fullfile(baseDir,imagingDir,subj_id, 'rmask_gray.nii');
%             dest = fullfile(baseDir,anatomicalDir,subj_id,'rmask_gray.nii');
%                 movefile(source,dest);            
    
    end 

end
