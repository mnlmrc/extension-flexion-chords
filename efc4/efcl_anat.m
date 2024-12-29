function varargout = efcl_anat(what, varargin)
    
    % Use a different baseDir when using your local machine or the cbs
    % server. Add more directory if needed.
    if isfolder('/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/')
        baseDir = '/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/';
    elseif isfolder("/path/to/project/cifs/directory/")
        baseDir = "/path/to/project/cifs/directory/";
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end

    bidsDir = 'BIDS'; % Raw data post AutoBids conversion
    anatomicalDir = 'anatomicals'; % anatomical files (individual space)
    freesurferDir = 'surfaceFreesurfer'; % freesurfer reconall output
    surfacewbDir = 'surfaceWB'; % fs32k template 
    suitDir = 'suit'; % SUIT 2.0 outputs
    
    pinfo = dload(fullfile(baseDir,'participants.tsv'));

    switch(what)
        case 'BIDS:move_unzip_raw_anat'
            % Moves, unzips and renames raw anatomical from 
            % BIDS directory. After you run this function you will find
            % an anatomical Nifti file named <subj_id>_T1w.nii in the 
            % <project_id>/anatomicals/<subj_id>/ directory.
            % This function assumes that the anatomical is in the first
            % session in the BIDS dir
            
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('BIDS:move_unzip_raw_anat -> ''sn'' must be passed to this function.')
            end
            
            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.participant_id{1};

%             % get the anatomical name
%             anat_name = subj_row.anat_name{1};
            
            % anatomical file
            anat_full_path = fullfile(baseDir,bidsDir,subj_id,'subj100/anat',sprintf('sub-%s_acq-MP2RAGE_run-01_T1w.nii.gz', sub_id));
            
           % Define output di
            output_folder = fullfile(baseDir,anatomicalDir, sub_id);
            dircheck(output_folder)
            output_file = fullfile(output_folder,sprintf('%s_T1w_raw.nii.gz', sub_id));
            
            % copy file to destination:
            copyfile(anat_full_path,output_file);
            
            % unzip the .gz files to make usable for SPM:
            gunzip(output_file);
            
            % delete the compressed file:
            delete(output_file);
        
        case 'ANAT:reslice_LPI'           % Reslice anatomical image within LPI coordinate systems
        % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:reslice_LPI -> ''sn'' must be passed to this function.')
            end
            
            subj_row=getrow(pinfo,pinfo.sn== s );
            subj_id = subj_row.participant_id{1};
            
            % (1) Reslice anatomical image to set it within LPI co-ordinate frames
            source  = fullfile(baseDir,anatomicalDir, subj_id, sprintf('%s_T1w_raw.nii', subj_id));
            
            dest    = fullfile(baseDir,anatomicalDir, subj_id,sprintf('%s_T1w_LPI.nii', subj_id));
            spmj_reslice_LPI(source,'name', dest);
            
            % (2) In the resliced image, set translation to zero
            V               = spm_vol(dest);
            dat             = spm_read_vols(V);
            V.mat(1:3,4)    = [0 0 0];
            spm_write_vol(V,dat);
            display 'Manually retrieve the location of the anterior commissure (x,y,z) before continuing'
        
        case 'ANAT:center_ac' % recenter to AC (manually retrieve coordinates)
            % Before running this step you need to manually fill in the AC
            % coordinates in the participants.tsv file
            % run spm display to get the AC coordinates
                
            sn=[];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:center_ac -> ''sn'' must be passed to this function.')
            end
            
            % get subj row from participants.tsv
            subj_row=getrow(pinfo,pinfo.sn== sn);
            subj_id = subj_row.participant_id{1};

            % Get the anat of subject
            subj_anat_img = fullfile(baseDir,anatomicalDir, sub_id, sprintf('%s_T1w_LPI.nii', subj_id));

            % get location of ac
            locACx = subj_id.locACx;
            locACy = subj_id.locACy;
            locACz = subj_id.locACz;

            loc_AC = [locACx locACy locACz];
            loc_AC = loc_AC';

            %recenter
            V               = spm_vol(subj_anat_img);
            dat             = spm_read_vols(V);
            oldOrig         = V.mat(1:3,4);
            V.mat(1:3,4)    = oldOrig-loc_AC;

            % Modify filename
            new_filename = fullfile(anatomical_dir, sub_id, sprintf('%s_T1w.nii', subj_id));
            V.fname = new_filename;
            spm_write_vol(V,dat);
                
        case 'ANAT:segment' % segment the anatomical image
            % check results when done
            
            sn=[];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:segment -> ''sn'' must be passed to this function.')
            end
            
            subj_row=getrow(pinfo,pinfo.sn== s );
            subj_id = subj_row.participant_id{1};
            
            subj_anat = fullfile(baseDir,anatomicalDir, sub_id, sprintf('%s_T1w.nii', subj_id);
    
            J.channel.vols     = {subj_anat};
            J.channel.biasreg  = 0.001;
            J.channel.biasfwhm = 60;
            J.channel.write    = [1 0];
            J.tissue(1).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,1')};
            J.tissue(1).ngaus  = 1;
            J.tissue(1).native = [1 0];
            J.tissue(1).warped = [0 0];
            J.tissue(2).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,2')};
            J.tissue(2).ngaus  = 1;
            J.tissue(2).native = [1 0];
            J.tissue(2).warped = [0 0];
            J.tissue(3).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,3')};
            J.tissue(3).ngaus  = 2;
            J.tissue(3).native = [1 0];
            J.tissue(3).warped = [0 0];
            J.tissue(4).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,4')};
            J.tissue(4).ngaus  = 3;
            J.tissue(4).native = [1 0];
            J.tissue(4).warped = [0 0];
            J.tissue(5).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,5')};
            J.tissue(5).ngaus  = 4;
            J.tissue(5).native = [1 0];
            J.tissue(5).warped = [0 0];
            J.tissue(6).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,6')};
            J.tissue(6).ngaus  = 2;
            J.tissue(6).native = [0 0];
            J.tissue(6).warped = [0 0];
    
            J.warp.mrf     = 1;
            J.warp.cleanup = 1;
            J.warp.reg     = [0 0.001 0.5 0.05 0.2];
            J.warp.affreg  = 'mni';
            J.warp.fwhm    = 0;
            J.warp.samp    = 3;
            J.warp.write   = [1 1];
            matlabbatch{1}.spm.spatial.preproc=J;
            spm_jobman('run',matlabbatch);
        
        case 'SURF:reconall' % Freesurfer reconall routine
            % Calls recon-all, which performs, all of the
            % FreeSurfer cortical reconstruction process
        
            sn=[];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('SURF:reconall -> ''sn'' must be passed to this function.')
            end

            subj_row=getrow(pinfo,pinfo.sn== s );
            subj_id = subj_row.participant_id{1};   
        
            % recon all inputs
            fs_dir = fullfile(baseDir,freesurferDir);
            anatomical_dir = fullfile(baseDir,anatomicalDir);
            anatomical_name = sprintf('%s_T1w.nii', subj_id);
            
            % Get the directory of subjects anatomical;
            freesurfer_reconall(fs_dir, sub_id, ...
                fullfile(anatomical_dir, sub_id, anatomical_name));
            
        case 'SURF:fs2wb'          % Resampling subject from freesurfer fsaverage to fs_LR        
            res  = 32;          % resolution of the atlas. options are: 32, 164
            hemi = [1, 2];      % list of hemispheres
            
            sn=[];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('SURF:fs2wb -> ''sn'' must be passed to this function.')
            end
    
            subj_row=getrow(pinfo,pinfo.sn== s );
            subj_id = subj_row.participant_id{1};  
            
            % get the subject id folder name
            outDir   = fullfile(baseDir, surfacewbDir; 
            dircheck(outDir);
            fs_dir = fullfile(baseDir,freesurferDir);
            surf_resliceFS2WB(sub_id, fs_dir, outDir, 'hemisphere', hemi, 'resolution', sprintf('%dk', res))
    
    end
            
    
    
    
end
    
