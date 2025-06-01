function varargout = efcl_anat(what, varargin)
    
    % Use a different baseDir when using your local machine or the cbs
    % server. Add more directory if needed.
    if isfolder('/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/')
        baseDir = '/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/';
        
        addpath(genpath('~/Documents/GitHub/dataframe/'))
        addpath(genpath('~/Documents/GitHub/spmj_tools/'))
        addpath(genpath('~/Documents/MATLAB/spm12/'))
        addpath(genpath('~/Documents/GitHub/surfAnalysis/'))
        
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
    regDir = 'ROI';
    wbDir = 'surfaceWB';
    glmEstDir = 'glm';
    
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
            anat_full_path = fullfile(baseDir,bidsDir,'day1',subj_id,  'anat',sprintf('sub-%d_acq-MP2RAGE_run-01_T1w.nii.gz', sn));
            
           % Define output di
            output_folder = fullfile(baseDir,anatomicalDir, subj_id);
%             dircheck(output_folder)
            output_file = fullfile(output_folder,sprintf('%s_T1w_raw.nii.gz', subj_id));
            
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
            
            subj_row=getrow(pinfo,pinfo.sn==sn);
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
        
        case 'ANAT:centre_AC' % recenter to AC (manually retrieve coordinates)
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
            subj_anat_img = fullfile(baseDir,anatomicalDir, subj_id, sprintf('%s_T1w_LPI.nii', subj_id));

            % get location of ac
            locACx = subj_row.locACx;
            locACy = subj_row.locACy;
            locACz = subj_row.locACz;

            loc_AC = [locACx locACy locACz];
            loc_AC = loc_AC';

%             %recenter
             V               = spm_vol(subj_anat_img);
             dat             = spm_read_vols(V);
%             oldOrig         = V.mat(1:3,4);
%             V.mat(1:3,4)    = oldOrig-loc_AC;
            
            R = V.mat(1:3,1:3);
            AC = [pinfo.locACx(pinfo.sn==sn),pinfo.locACy(pinfo.sn==sn),pinfo.locACz(pinfo.sn==sn)]';
            t = -1 * R * AC;
            V.mat(1:3,4) = t;
            sprintf('ACx: %d, ACy: %d, ACz: %d', pinfo.locACx(pinfo.sn==sn), pinfo.locACy(pinfo.sn==sn), pinfo.locACz(pinfo.sn==sn))

            % Modify filename
            new_filename = fullfile(baseDir, anatomicalDir, subj_id, sprintf('%s_T1w.nii', subj_id));
            V.fname = new_filename;
            spm_write_vol(V,dat);
                
        case 'ANAT:segment' % segment the anatomical image
            % check results when done
            
            sn=[];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:segment -> ''sn'' must be passed to this function.')
            end
            
            subj_row=getrow(pinfo,pinfo.sn== sn );
            subj_id = subj_row.participant_id{1};
            
            subj_anat = fullfile(baseDir,anatomicalDir, subj_id, sprintf('%s_T1w.nii', subj_id));
            
            SPMhome=fileparts(which('spm.m'));
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

            subj_row=getrow(pinfo,pinfo.sn== sn );
            subj_id = subj_row.participant_id{1};   
        
            % recon all inputs
            fs_dir = fullfile(baseDir,freesurferDir, subj_id);

            if ~exist(fs_dir,"dir")
                mkdir(fs_dir);
            end

            anatomical_dir = fullfile(baseDir,anatomicalDir);
            anatomical_name = sprintf('%s_T1w.nii', subj_id);
            
            % Get the directory of subjects anatomical;
            freesurfer_reconall(fs_dir, subj_id, ...
                fullfile(anatomical_dir, subj_id, anatomical_name));
            
        case 'SURF:fs2wb'          % Resampling subject from freesurfer fsaverage to fs_LR        
            res  = 32;          % resolution of the atlas. options are: 32, 164
            hemi = [1, 2];      % list of hemispheres
            
            sn=[];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('SURF:fs2wb -> ''sn'' must be passed to this function.')
            end
    
            subj_row=getrow(pinfo,pinfo.sn== sn );
            subj_id = subj_row.participant_id{1};  
            
            % get the subject id folder name
            outDir   = fullfile(baseDir, surfacewbDir); 
%             dircheck(outDir);
            fs_dir = fullfile(baseDir,freesurferDir, subj_id);
            surf_resliceFS2WB(subj_id, fs_dir, outDir, 'hemisphere', hemi, 'resolution', sprintf('%dk', res))
        
        case 'ROI:define'
            
            sn = [];
            glm = 1;
            atlas = 'ROI';
            
            vararginoptions(varargin,{'sn', 'atlas', 'glm'});
            
            if isfolder('/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32')
                atlasDir = '/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32';
            elseif isfolder('/cifs/diedrichsen/data/Atlas_templates/fs_LR_32')
                atlasDir = '/cifs/diedrichsen/data/Atlas_templates/fs_LR_32';
            end
            atlasH = {sprintf('%s.32k.L.label.gii', atlas), sprintf('%s.32k.R.label.gii', atlas)};
            atlas_gii = {gifti(fullfile(atlasDir, atlasH{1})), gifti(fullfile(atlasDir, atlasH{1}))};

            subj_row=getrow(pinfo,pinfo.sn==sn);
            subj_id = subj_row.participant_id{1};

            Hem = {'L', 'R'};
            R = {};
            r = 1;
            for h = 1:length(Hem)
                for reg = 1:length(atlas_gii{h}.labels.name)

                    R{r}.white = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.white.32k.surf.gii']);
                    R{r}.pial = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.pial.32k.surf.gii']);
                    R{r}.image = fullfile(baseDir, [glmEstDir '1'], 'day1', subj_id, 'mask.nii');
                    R{r}.linedef = [5 0 1];
                    key = atlas_gii{h}.labels.key(reg);
                    R{r}.location = find(atlas_gii{h}.cdata==key);
                    R{r}.hem = Hem{h};
                    R{r}.name = atlas_gii{h}.labels.name{reg};
                    R{r}.type = 'surf_nodes_wb';

                    r = r+1;
                end
            end

            R = region_calcregions(R, 'exclude', [2 3; 2 4; 2 5; 4 5; 8 9; 2 8;...
                11 12; 11 13; 11 14; 13 14; 17 18; 11 17], 'exclude_thres', .8);
            
            output_path = fullfile(baseDir, regDir, 'day1', subj_id);
            if ~exist(output_path, 'dir')
                mkdir(output_path)
            end
            
            Vol = fullfile(baseDir, [glmEstDir '1'], 'day1', subj_id, 'mask.nii');
            for r = 1:length(R)
                img = region_saveasimg(R{r}, Vol, 'name',fullfile(baseDir, regDir, 'day1', subj_id, sprintf('%s.%s.%s.nii', atlas, R{r}.hem, R{r}.name)));
            end       
            
            save(fullfile(output_path, sprintf('%s_%s_region.mat',subj_id, atlas)), 'R');

    
    end
            
    
    
    
end
    
