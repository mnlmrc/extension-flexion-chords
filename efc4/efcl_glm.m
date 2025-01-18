function varargout = efcl_glm(what, varargin)

    % Use a different baseDir when using your local machine or the cbs
    % server. Add more directory if needed. Use single quotes ' and not
    % double quotes " because some spm function raise error with double
    % quotes
    if isfolder('/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/')
        baseDir = '/cifs/diedrichsen/data/Chord_exp/ExtFlexChord/efc4/';
        
        addpath(genpath('~/Documents/GitHub/dataframe/'))
        addpath(genpath('~/Documents/GitHub/spmj_tools/'))
        addpath(genpath('~/Documents/GitHub/rwls/'))
        addpath(genpath('~/Documents/MATLAB/spm12/'))
        addpath(genpath('~/Documents/GitHub/surfAnalysis/'))
        addpath(genpath('~/Documents/GitHub/surfing/surfing'))
        
        
    elseif isfolder('/path/to/project/cifs/directory/')
        baseDir = '/path/to/project/cifs/directory/';
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end

    sn = [];
    day = [];
    glm = [];
    type = 'spmT';
    hrf_params = [5 14 1 1 6 0 32];
    derivs = [0, 0];
    vararginoptions(varargin,{'sn', 'day', 'type', 'glm', 'hrf_params', 'derivs'})

    glmEstDir = 'glm';
    behavDir = 'behavioural';
    % anatomicalDir = 'anatomicals';
    imagingDir = 'imaging_data';
    wbDir = 'surfaceWB';

    pinfo = dload(fullfile(baseDir,'participants.tsv'));

    % get participant row from participant.tsv
    subj_row=getrow(pinfo, pinfo.sn== sn);
    
    % get subj_id
    subj_id = subj_row.participant_id{1};
    
    % get day_id
    day_id = sprintf('day%d', day);

    % get runs (FuncRuns column needs to be in participants.tsv)    
    runs = spmj_dotstr2array(subj_row.FuncRuns{1});

    switch what
        case 'GLM:make_glm1'

            D = dload(fullfile(baseDir, behavDir, day_id, subj_id, sprintf('efc4_%d.dat', sn)));

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.eventtype = [];
            
            for chordID = chords'
                
                events.BN = [events.BN; D.BN(D.chordID == chordID)];
                events.TN = [events.TN; D.TN(D.chordID == chordID)];
                events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID) + 500];
                events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID)];
                events.eventtype = [events.eventtype; D.chordID(D.chordID == chordID)];
                
            end
            
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            varargout{1} = events;

        case 'GLM:make_glm2'

            D = dload(fullfile(baseDir, behavDir, day_id, subj_id, sprintf('efc4_%d.dat', sn)));

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.eventtype = [];
            
            for chordID = chords'
                
                events.BN = [events.BN; D.BN(D.chordID == chordID & D.trialPoint == 1)];
                events.TN = [events.TN; D.TN(D.chordID == chordID & D.trialPoint == 1)];
                events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.trialPoint == 1) + 500];
                events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID & D.trialPoint == 1)];
                events.eventtype = [events.eventtype; D.chordID(D.chordID == chordID & D.trialPoint == 1)];
                
            end
            
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            varargout{1} = events;
            
        case 'GLM:make_glm3'

            D = dload(fullfile(baseDir, behavDir, day_id, subj_id, sprintf('efc4_%d.dat', sn)));
            
            D.repetition = ones(length(D.TN), 1); % Initialize repetition column with 1

            for i = 1:length(D.TN)
                if i == 1
                    D.repetition(i) = 1;
                else
                    if D.chordID(i) == D.chordID(i-1)
                        D.repetition(i) = 2;
                    end
                end
            end
    
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.eventtype = [];
            
            for rep = unique(D.repetition)'
                for chordID = unique(D.chordID)'

                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.repetition == rep)];
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.repetition == rep)];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.repetition == rep) + 500];
                    events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID & D.repetition == rep)];
                    events.eventtype = [events.eventtype; D.chordID(D.chordID == chordID & D.repetition == rep)];

                end
            end
            
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            varargout{1} = events;
            
        case 'GLM:make_event'
    
            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(subj_row.FuncRuns{1});
            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end 
            
            operation  = sprintf('GLM:make_glm%d', glm);
            
            events = efcl_glm(operation, 'sn', sn, 'day', day);
            events = events(ismember(events.BN, runs), :);
            
            %% export
            output_folder = fullfile(baseDir, behavDir, day_id, subj_id);
            writetable(events, fullfile(output_folder, sprintf('glm%d_events.tsv', glm)), 'FileType', 'text', 'Delimiter','\t')

            if ~isfolder(fullfile(baseDir, [glmEstDir num2str(glm)] , subj_id))
                mkdir(fullfile(baseDir, [glmEstDir num2str(glm)], subj_id))
            end
            
        case 'GLM:design'

            % Import globals from spm_defaults 
            global defaults; 
            if (isempty(defaults)) 
                spm_defaults;
            end 
            
            currentDir = pwd;

            if isempty(sn)
                error('GLM:design -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:design -> ''glm'' must be passed to this function.')
            end

            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end

            % Load data once, outside of session loop
            % D = dload(fullfile(baseDir,behavDir,subj_id, sprintf('smp2_%d.dat', sn)));
            events_file = sprintf('glm%d_events.tsv', glm);

            Dd = dload(fullfile(baseDir,behavDir, day_id, subj_id, events_file));
            Dd.eventtype = cellstr(string(Dd.eventtype));
            
            regressors = unique(Dd.eventtype);
            nRegr = length(regressors); 

            % init J
            J = [];
            T = [];
            J.dir = {fullfile(baseDir,sprintf('glm%d', glm),day_id, subj_id)};
            J.timing.units = 'secs';
            J.timing.RT = 1;

            % number of temporal bins in which the TR is divided,
            % defines the discrtization of the HRF inside each TR
            J.timing.fmri_t = 16;

            % slice number that corresponds to that acquired halfway in
            % each TR
            J.timing.fmri_t0 = 1;
        
            for run = runs
                % Setup scans for current session
                J.sess(run).scans = {fullfile(baseDir, imagingDir, day_id, subj_id, sprintf('u%s_run_%02d.nii', subj_id, run))};
        
        
                % Preallocate memory for conditions
                J.sess(run).cond = repmat(struct('name', '', 'onset', [], 'duration', []), nRegr, 1);
                
                for regr = 1:nRegr
                    % cue = Dd.cue(regr);
                    % stimFinger = Dd.stimFinger(regr);
                    rows = find(Dd.BN == run & strcmp(Dd.eventtype, regressors(regr)));
                    % cue_id = unique(Dd.cue_id(rows));
                    % stimFinger_id = unique(Dd.stimFinger_id(rows));
                    % epoch = unique(Dd.epoch(rows));
                    % instr = unique(Dd.instruction(rows));
                    
                    % Regressor name
                    J.sess(run).cond(regr).name = regressors{regr};
                    
                    % Define durationDuration(regr));
                    J.sess(run).cond(regr).duration = Dd.Duration(rows); % needs to be in seconds
                    
                    % Define onset
                    J.sess(run).cond(regr).onset  = Dd.Onset(rows);
                    
                    % Define time modulator
                    % Add a regressor that account for modulation of
                    % betas over time
                    J.sess(run).cond(regr).tmod = 0;
                    
                    % Orthogonalize parametric modulator
                    % Make the parametric modulator orthogonal to the
                    % main regressor
                    J.sess(run).cond(regr).orth = 0;
                    
                    % Define parametric modulators
                    % Add a parametric modulators, like force or
                    % reaction time. 
                    J.sess(run).cond(regr).pmod = struct('name', {}, 'param', {}, 'poly', {});

                    %
                    % filling in "reginfo"
                    TT.sn        = sn;
                    TT.run       = run;
                    TT.name      = regressors(regr);
                    % TT.cue       = cue_id;
                    % TT.epoch     = epoch;
                    % TT.stimFinger = stimFinger_id;
                    % TT.instr = instr;       

                    T = addstruct(T, TT);

                end

                % Specify high pass filter
                J.sess(run).hpf = Inf;

                % J.sess(run).multi
                % Purpose: Specifies multiple conditions for a session. Usage: It is used
                % to point to a file (.mat or .txt) that contains multiple conditions,
                % their onsets, durations, and names in a structured format. If you have a
                % complex design where specifying conditions manually within the script is
                % cumbersome, you can prepare this information in advance and just
                % reference the file here. Example Setting: J.sess(run).multi =
                % {'path/to/multiple_conditions_file.mat'}; If set to {' '}, it indicates
                % that you are not using an external file to specify multiple conditions,
                % and you will define conditions directly in the script (as seen with
                % J.sess(run).cond).
                J.sess(run).multi     = {''};                        

                % J.sess(run).regress
                % Purpose: Allows you to specify additional regressors that are not
                % explicitly modeled as part of the experimental design but may account for
                % observed variations in the BOLD signal. Usage: This could include
                % physiological measurements (like heart rate or respiration) or other
                % variables of interest. Each regressor has a name and a vector of values
                % corresponding to each scan/time point.
                J.sess(run).regress   = struct('name', {}, 'val', {});                        

                % J.sess(run).multi_reg Purpose: Specifies a file containing multiple
                % regressors that will be included in the model as covariates. Usage: This
                % is often used for motion correction, where the motion parameters
                % estimated during preprocessing are included as regressors to account for
                % motion-related artifacts in the BOLD signal. Example Setting:
                % J.sess(run).multi_reg = {'path/to/motion_parameters.txt'}; The file
                % should contain a matrix with as many columns as there are regressors and
                % as many rows as there are scans/time points. Each column represents a
                % different regressor (e.g., the six motion parameters from realignment),
                % and each row corresponds to the value of those regressors at each scan.
                J.sess(run).multi_reg = {''};
                
                % Specify factorial design
                J.fact             = struct('name', {}, 'levels', {});

                % Specify hrf parameters for convolution with
                % regressors
                J.bases.hrf.derivs = derivs;
                J.bases.hrf.params = hrf_params;  % positive and negative peak of HRF - set to [] if running wls (?)
                defaults.stats.fmri.hrf=J.bases.hrf.params; 
                
                % Specify the order of the Volterra series expansion 
                % for modeling nonlinear interactions in the BOLD response
                % *Example Usage*: Most analyses use 1, assuming a linear
                % relationship between neural activity and the BOLD
                % signal.
                J.volt = 1;

                % Specifies the method for global normalization, which
                % is a step to account for global differences in signal
                % intensity across the entire brain or between scans.
                J.global = 'None';

                % remove voxels involving non-neural tissue (e.g., skull)
                J.mask = {fullfile(baseDir, imagingDir, day_id,subj_id,  'rmask_noskull.nii')};
                
                % Set threshold for brightness threshold for masking 
                % If supplying explicit mask, set to 0  (default is 0.8)
                J.mthresh = 0.;

                % Create map where non-sphericity correction must be
                % applied
                J.cvi_mask = {fullfile(baseDir, imagingDir, day_id, subj_id,  'rmask_gray.nii')};

                % Method for non sphericity correction
                J.cvi =  'fast';
                
            end

            % T.cue000 = strcmp(T.cue, 'cue0');
            % T.cue025 = strcmp(T.cue, 'cue25');
            % T.cue050 = strcmp(T.cue, 'cue50');
            % T.cue075 = strcmp(T.cue, 'cue75');
            % T.cue100 = strcmp(T.cue, 'cue100');
            % 
            % T.index = strcmp(T.stimFinger, 'index');
            % T.ring = strcmp(T.stimFinger, 'ring');
            % 
            % T.plan = strcmp(T.epoch, 'plan');
            % T.exec = strcmp(T.epoch, 'exec');
            % 
            % T.go = strcmp(T.instr, 'go');
            % T.nogo = strcmp(T.instr, 'nogo');
            % 
            % T.rest = strcmp(T.name, 'rest');


            % remove empty rows (e.g., when skipping runs)
            J.sess = J.sess(~arrayfun(@(x) all(structfun(@isempty, x)), J.sess));
            
            if ~exist(J.dir{1},"dir")
                mkdir(J.dir{1});
            end
            
            dsave(fullfile(J.dir{1},sprintf('%s_%s_reginfo.tsv', day_id, subj_id)), T);
            spm_rwls_run_fmri_spec(J);

            cd(currentDir)
            
            currentDir = pwd;
            
       case 'GLM:estimate'      % estimate beta values

            currentDir = pwd;

            if isempty(sn)
                error('GLM:estimate -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:estimate -> ''glm'' must be passed to this function.')
            end

%             fprintf('- Doing glm%d estimation for subj %s\n', glm, day_id, subj_id);
            subj_est_dir = fullfile(baseDir, sprintf('glm%d', glm), day_id, subj_id);                
            SPM = load(fullfile(subj_est_dir,'SPM.mat'));
            SPM.SPM.swd = subj_est_dir;

            iB = SPM.SPM.xX.iB;

            save(fullfile(subj_est_dir, "iB.mat"), "iB");

            spm_rwls_spm(SPM.SPM);

            cd(currentDir)
            
       case 'GLM:T_contrasts'
            
            currentDir = pwd;

            replace_xCon   = true;

            if isempty(sn)
                error('GLM:T_contrasts -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:T_contrasts -> ''glm'' must be passed to this function.')
            end

            % get the subject id folder name
            fprintf('Contrasts for participant %s\n', subj_id)
            glm_dir = fullfile(baseDir, sprintf('glm%d', glm), day_id, subj_id); 

            % load the SPM.mat file
            SPM = load(fullfile(glm_dir, 'SPM.mat')); SPM=SPM.SPM;

            if replace_xCon
                SPM  = rmfield(SPM,'xCon');
            end

            T    = dload(fullfile(glm_dir, sprintf('%s_%s_reginfo.tsv', day_id, subj_id)));
            T.name = cellstr(string(T.name));
            contrasts = unique(T.name);

            for c = 1:length(contrasts)
 
                contrast_name = contrasts{c};
                xcon = zeros(size(SPM.xX.X,2), 1);
                xcon(strcmp(T.name, contrast_name)) = 1;
                xcon = xcon / sum(xcon);
                if ~isfield(SPM, 'xCon')
                    SPM.xCon = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = 1;
                elseif sum(strcmp(contrast_name, {SPM.xCon.name})) > 0
                    idx = find(strcmp(contrast_name, {SPM.xCon.name}));
                    SPM.xCon(idx) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = idx;
                else
                    SPM.xCon(end+1) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = length(SPM.xCon);
                end
                SPM = spm_contrasts(SPM,1:length(SPM.xCon));
                save('SPM.mat', 'SPM','-v7.3');
                % SPM = rmfield(SPM,'xVi'); % 'xVi' take up a lot of space and slows down code!
                % save(fullfile(glm_dir, 'SPM_light.mat'), 'SPM')
    
                % rename contrast images and spmT images
                conName = {'con','spmT'};
                for n = 1:numel(conName)
                    oldName = fullfile(glm_dir, sprintf('%s_%2.4d.nii',conName{n},cname_idx));
                    newName = fullfile(glm_dir, sprintf('%s_%s.nii',conName{n},SPM.xCon(cname_idx).name));
                    movefile(oldName, newName);
                end % conditions (n, conName: con and spmT)

            end

            cd(currentDir)
            
       case 'GLM:all'
            
            spm_get_defaults('cmdline', true);  % Suppress GUI prompts, no request for overwirte

                
            % Check for and delete existing SPM.mat file
            spm_file = fullfile(baseDir, [glmEstDir num2str(glm)], ['subj' num2str(sn)], 'SPM.mat');
            if exist(spm_file, 'file')
                delete(spm_file);
            end

            efcl_glm('GLM:make_event', 'sn', sn, 'glm', glm, 'day', day)
            efcl_glm('GLM:design', 'sn', sn, 'glm', glm, 'hrf_params', hrf_params, 'day', day, 'derivs', derivs)
            efcl_glm('GLM:estimate', 'sn', sn, 'glm', glm, 'day', day)
            efcl_glm('GLM:T_contrasts', 'sn', sn, 'glm', glm, 'day', day)
            efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'spmT', 'day', day)
%             efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'beta')
%             efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'res')
%             efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'con')
%             efcl_glm('HRF:ROI_hrf_get', 'sn', sn, 'glm', glm, 'hrf_params', hrf_params)
            
       case 'SURF:vol2surf'
            
            currentDir = pwd;

            res  = 32;          % resolution of the atlas. options are: 32, 164

            glmEstDir = [glmEstDir num2str(glm)];
            
            V = {};
            cols = {};
            if strcmp(type, 'spmT')
%                 filename = ['spmT_' id '.func.gii'];
                files = dir(fullfile(baseDir, glmEstDir, day_id, subj_id, 'spmT_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'beta')
                SPM = load(fullfile(baseDir, glmEstDir, day_id,subj_id, 'SPM.mat')); SPM=SPM.SPM;
                files = dir(fullfile(baseDir, glmEstDir, day_id,subj_id, 'beta_*.nii'));
                files = files(SPM.xX.iC);
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'psc')
                files = dir(fullfile(baseDir, glmEstDir, day_id,subj_id, 'psc_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'con')
                files = dir(fullfile(baseDir, glmEstDir, day_id,subj_id, 'con_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'res')
                V{1} = fullfile(baseDir, glmEstDir,day_id, subj_id, 'ResMS.nii');
                cols{1} = 'ResMS';
            end

            hemLpial = fullfile(baseDir, wbDir, subj_id,  [subj_id '.L.pial.32k.surf.gii']);
            hemRpial = fullfile(baseDir, wbDir, subj_id, [subj_id '.R.pial.32k.surf.gii']);
            hemLwhite = fullfile(baseDir, wbDir, subj_id, [subj_id '.L.white.32k.surf.gii']);
            hemRwhite = fullfile(baseDir, wbDir, subj_id, [subj_id '.R.white.32k.surf.gii']);
            
            hemLpial = gifti(hemLpial);
            hemRpial = gifti(hemRpial);
            hemLwhite = gifti(hemLwhite);
            hemRwhite = gifti(hemRwhite);

            c1L = hemLpial.vertices;
            c2L = hemLwhite.vertices;
            c1R = hemRpial.vertices;
            c2R = hemRwhite.vertices;

            GL = surf_vol2surf(c1L,c2L,V,'anatomicalStruct','CortexLeft', 'exclude_thres', 0.9, 'faces', hemLpial.faces);
            GL = surf_makeFuncGifti(GL.cdata,'anatomicalStruct', 'CortexLeft', 'columnNames', cols);
    
            save(GL, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.' day_id '.'  type '.L.func.gii']))
    
            GR = surf_vol2surf(c1R,c2R,V,'anatomicalStruct','CortexRight', 'exclude_thres', 0.9, 'faces', hemRpial.faces);
            GR = surf_makeFuncGifti(GR.cdata,'anatomicalStruct', 'CortexRight', 'columnNames', cols);

            save(GR, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.' day_id '.' type '.R.func.gii']))
            
            cd(currentDir)

            
    end

end