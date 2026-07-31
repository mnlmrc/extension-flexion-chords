function varargout = efcl_glm(what, varargin)

    % Use a different baseDir when using your local machine or the cbs
    % server. Add more directory if needed. Use single quotes ' and not
    % double quotes " because some spm function raise error with double
    % quotes
    if isfolder('/cifs/diedrichsen/data/Chord_exp/EFC_learningfMRI/')
        baseDir = '/cifs/diedrichsen/data/Chord_exp/EFC_learningfMRI/';
        
        addpath(genpath('~/Documents/GitHub/dataframe/'))
        addpath(genpath('~/Documents/GitHub/spmj_tools/'))
        addpath(genpath('~/Documents/GitHub/rwls/'))
        addpath(genpath('~/Documents/MATLAB/spm12/'))
        addpath(genpath('~/Documents/GitHub/surfAnalysis/'))
        addpath(genpath('~/Documents/GitHub/surfing/surfing'))
        addpath(genpath('~/Documents/GitHub/region/'))
        
        
    elseif isfolder('/path/to/project/cifs/directory/')
        baseDir = '/path/to/project/cifs/directory/';
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end

    sn = [];
    day = [];
    glm = [];
    type = 'spmT';
    fit_to_day = 'common';
    derivs = [0, 0];
    vararginoptions(varargin,{'sn', 'day', 'type', 'glm', 'hrf_params', 'atlas','derivs', 'fit_to_day'})

    %glmEstDir = 'glm';
    behavDir = 'behavioural';
    localDir = '/localscratch/tmp/SMP';
    imagingDir = 'imaging_data';
    wbDir = 'surfaceWB';
    
    if isscalar(sn)
        pinfo = dload(fullfile(baseDir,'participants.tsv'));

        % get participant row from participant.tsv
        subj_row=getrow(pinfo, pinfo.sn== sn);
        
        % get subj_id
        subj_id = subj_row.participant_id{1};
        
        D = dload(fullfile(baseDir, behavDir, sprintf('day%d', day(1)), sprintf('efc4_%d.dat', sn)));
        if length(day) > 1
            for i=2:length(day)
                D_tmp = dload(fullfile(baseDir, behavDir, sprintf('day%d', day(i)), sprintf('efc4_%d.dat', sn)));
                fields = fieldnames(D_tmp);
                for j = 1:numel(fields)
                    fname = fields{j};
                    if isfield(D, fname)
                        D.(fname) = [D.(fname); D_tmp.(fname)];
                    end
                end
            end
        end

        if isscalar(day)
            subj_est_dir = fullfile(baseDir, sprintf('glm%d',  glm), subj_id, sprintf('day%d',  day));
        else
            subj_est_dir = fullfile(baseDir, sprintf('glm%d',  glm), subj_id);
        end
    
        % get runs (FuncRuns column needs to be in participants.tsv)
        if isscalar(day)
            if day == 3
                d = 0;
            elseif day == 9
                d =1;
            elseif day == 23
                d=2;
            end
            ref_day = sprintf('day%d', day); % day where to save the events file
            runs = spmj_dotstr2array(subj_row.(sprintf('FuncRuns_day%d', day)){1});
        else
            runs = [];
            ref_day = 'day3'; % day where to save the events file
            for i = 1:length(day)
                if day(i) == 3
                    d = 0;
                elseif day(i) == 9
                    d =1;
                elseif day(i) == 23
                    d=2;
                end
                % day_id{i} = sprintf('day%d', day{i});
                runs = [runs spmj_dotstr2array(subj_row.(sprintf('FuncRuns_day%d', day(i))){1}) + 10*d];
            end
        end
    end

    switch what
       case 'GLM:make_glm1'

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.chordID = [];
            events.day = [];
            events.eventtype = [];
            
            for d = 1:length(day)
                for chordID = chords'
                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d)) + 10 * (d - 1)] ;
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d))];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d)) + 1000];
                    events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID & D.day == day(d))];
                    n_rep = length(D.BN(D.chordID == chordID & D.day == day(d)));
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d', chordID, day(d))}, [n_rep, 1])];
                end
            end
            
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            varargout{1} = events;

       case 'GLM:make_glm2' % ignore day

            chords = unique(D.chordID);
            D.repetition = mod(D.TN - 1, 2) + 1;
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.chordID = [];
            events.day = [];
            events.repetition = [];
            events.eventtype = [];
            
            for rep = 1:2
                for d = 1:length(day)
                    for chordID = chords'
                        events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d) & D.repetition == rep) + 10 * (d - 1)] ;
                        events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d) & D.repetition == rep)];
                        events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d) & D.repetition == rep) + 1000];
                        %events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID & D.day == day(d) & D.repetition == rep)];
                        n_rep = length(D.BN(D.chordID == chordID & D.day == day(d) & D.repetition == rep));
                        events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                        events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                        events.repetition = [events.repetition; repmat({sprintf('%d', rep)}, [n_rep, 1])];
                        events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d,%d', chordID, day(d), rep)}, [n_rep, 1])];
                    end
                end
            end
            
            events.Duration = zeros(length(events.BN), 1);
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            
            varargout{1} = events;

       case 'GLM:make_glm3'

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.chordID = [];
            events.day = [];
            events.eventtype = [];
            
            for d = 1:length(day)
                for chordID = chords'
                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d)) + 10 * (d - 1)] ;
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d))];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d)) + 1000];
                    n_rep = length(D.BN(D.chordID == chordID & D.day == day(d)));
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d', chordID, day(d))}, [n_rep, 1])];
                end
            end
            
            events.Duration = zeros(length(events.BN), 1);
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            
            varargout{1} = events;

       case 'GLM:make_glm6'

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.chordID = [];
            events.day = [];
            events.eventtype = [];
            
            for d = 1:length(day)
                for chordID = chords'
                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d)) + 10 * (d - 1)] ;
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d))];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d)) + 1000];
                    n_rep = length(D.BN(D.chordID == chordID & D.day == day(d)));
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d', chordID, day(d))}, [n_rep, 1])];
                end
            end
            
            events.Duration = zeros(length(events.BN), 1) + 1; % BOXCAR OF 1s
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            
            varargout{1} = events;

       case 'GLM:make_glm7'

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.chordID = [];
            events.day = [];
            events.eventtype = [];
            
            for d = 1:length(day)
                for chordID = chords'
                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d)) + 10 * (d - 1)] ;
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d))];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d))];
                    n_rep = length(D.BN(D.chordID == chordID & D.day == day(d)));
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d,plan', chordID, day(d))}, [n_rep, 1])];
                end
            end

            for d = 1:length(day)
                for chordID = chords'
                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d)) + 10 * (d - 1)] ;
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d))];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d)) + 1000];
                    n_rep = length(D.BN(D.chordID == chordID & D.day == day(d)));
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d,exec', chordID, day(d))}, [n_rep, 1])];
                end
            end
            
            events.Duration = zeros(length(events.BN), 1) + 1; % BOXCAR OF 1s
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            
            varargout{1} = events;
       
       case 'GLM:make_glm4'

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.chordID = [];
            events.day = [];
            events.eventtype = [];
            
            for d = 1:length(day)
                for chordID = chords'
                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d))] ;
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d))];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d)) + 1000];
                    events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID & D.day == day(d))];
                    n_rep = length(D.BN(D.chordID == chordID & D.day == day(d)));
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d', chordID, day(d))}, [n_rep, 1])];
                end
            end
            
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            varargout{1} = events;

       case 'GLM:make_glm5' 

            chords = unique(D.chordID);
            
            events.BN = [];
            events.TN = [];
            events.Onset = [];
            events.Duration = [];
            events.chordID = [];
            events.day = [];
            events.eventtype = [];
            
            for d = 1:length(day)
                for chordID = chords'
                    events.BN = [events.BN; D.BN(D.chordID == chordID & D.day == day(d)) + 10 * (d - 1)] ;
                    events.TN = [events.TN; D.TN(D.chordID == chordID & D.day == day(d))];
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d)) + 1000];
                    events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID & D.day == day(d))];
                    n_rep = length(D.BN(D.chordID == chordID & D.day == day(d)));
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [n_rep, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,sess%02d', chordID, day(d))}, [n_rep, 1])];
                end
                
                events.BN = [events.BN; D.BN(D.day == day(d)) + 10 * (d - 1)] ;
                events.TN = [events.TN; D.TN(D.day == day(d))];
                events.Onset = [events.Onset; D.startTimeReal( D.day == day(d)) + 4500];
                n_rep = length(D.BN(D.day == day(d)));
                events.Duration = [events.Duration; repmat(0, [n_rep, 1])];
                events.chordID = [events.chordID; repmat({99999}, [n_rep, 1])];
                events.day = [events.day; repmat({sprintf('%d', day(d))}, [n_rep, 1])];
                events.eventtype = [events.eventtype; repmat({sprintf('99999,sess%02d', day(d))}, [n_rep, 1])];

            end

            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            varargout{1} = events;

       case 'GLM:make_event'
            
            operation  = sprintf('GLM:make_glm%d', glm);
            
            events = efcl_glm(operation, 'sn', sn, 'day', day);
            events = events(ismember(events.BN, runs), :);
            
            output_folder = fullfile(baseDir, behavDir, ref_day);
            writetable(events, fullfile(output_folder, sprintf('efc4_%s_glm%d_events.tsv', subj_id, glm)), 'FileType', 'text', 'Delimiter','\t')
            
       case 'GLM:design'

            % Import globals from spm_defaults 
            global defaults;
            if (isempty(defaults)) 
                spm_defaults;
            end
            defaults.mat.format = '-v7.3';
            %defaults.stats.maxmem = 2^26;
            
            currentDir = pwd;

            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end

            % Load data once, outside of session loop
            % D = dload(fullfile(baseDir,behavDir,subj_id, sprintf('smp2_%d.dat', sn)));
            events_file = sprintf('efc4_%s_glm%d_events.tsv', subj_id, glm);

            Dd = dload(fullfile(baseDir, behavDir, ref_day, events_file));
            %eventtype = unique(Dd.eventtype);
            
            regressors = unique(Dd.eventtype);
            nRegr = length(regressors); 

            % init J
            J = [];
            T = [];
            J.dir = {localDir}; % {subj_est_dir};
            J.timing.units = 'secs';
            J.timing.RT = 1;

            % number of temporal bins in which the TR is divided,
            % defines the discrtization of the HRF inside each TR
            J.timing.fmri_t = 16;

            % slice number that corresponds to that acquired halfway in
            % each TR
            J.timing.fmri_t0 = 1;
            
            for run = runs
                
                run_local = run - min(runs) + 1;

                % Setup scans for current session
                J.sess(run_local).scans = {fullfile(baseDir, imagingDir, subj_id, sprintf('u%s_run_%02d.nii', subj_id, run))};
        
                % Preallocate memory for conditions
                J.sess(run_local).cond = repmat(struct('name', '', 'onset', [], 'duration', []), nRegr/length(day), 1);

                regr = 1;
                for regressor = regressors'
                    rows = find(Dd.BN == run & strcmp(Dd.eventtype, regressor{1}));
                    %rows = find(Dd.BN == run & Dd.day == day(d) & Dd.chordID == chordID);

                    if ~isempty(rows)
                        % Regressor name
                        %J.sess(run).cond(regr).name = sprintf('%d,%d', day(d), chordID);
                        J.sess(run).cond(regr).name = regressor{1};
                        
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
    
                        % filling in "reginfo"
                        TT.sn        = sn;
                        TT.run       = run; %mod(run - 1, 10) + 1;
                        TT.name      = regressor{1}; % sprintf('%02d,%d', day(d), chordID);      
    
                        T = addstruct(T, TT);

                        regr = regr + 1;
                
                    end
                end

                % Specify high pass filter
                J.sess(run).hpf = 128; % 128; %256;

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
                J.mask = {fullfile(baseDir, imagingDir,subj_id,  'rmask_noskull.nii')};
                
                % Set threshold for brightness threshold for masking 
                % If supplying explicit mask, set to 0  (default is 0.8)
                J.mthresh = 0.;

                % Create map where non-sphericity correction must be
                % applied
                J.cvi_mask = {fullfile(baseDir, imagingDir, subj_id,  'rmask_gray.nii')};

                % Method for non sphericity correction
                J.cvi =  'fast';
                
            end

            % remove empty rows (e.g., when skipping runs)
            J.sess = J.sess(~arrayfun(@(x) all(structfun(@isempty, x)), J.sess));
            
            if ~exist(J.dir{1},"dir")
                mkdir(J.dir{1});
            end
            
            dsave(fullfile(J.dir{1},'reginfo.tsv'), T);

            defaults.mat.format = '-v7.3';
            spm_rwls_run_fmri_spec(J);

            cd(currentDir)
            
       case 'GLM:estimate'      % estimate beta values

            % Import globals from spm_defaults 
            global defaults;
            if (isempty(defaults)) 
                spm_defaults;
            end 

            defaults.mat.format = '-v7.3';
            gib = 6; % Gib used to estimate GLM reduce if code crashes
            spm_get_defaults('stats.maxmem', gib*1024^3);  
            %defaults.stats.maxmem = 16*1024^3;

            currentDir = pwd;

            % fprintf('- Doing glm%d estimation for subj %s\n', glm, day_id, subj_id);
            SPM = load(fullfile(localDir,'SPM.mat'));

            % if exist(localDir,'dir'); rmdir(localDir,'s'); end
            % mkdir(localDir);
            SPM.SPM.swd = localDir;

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

            % load the SPM.mat file
            SPM = load(fullfile(localDir, 'SPM.mat')); SPM=SPM.SPM;

            if replace_xCon
                SPM  = rmfield(SPM,'xCon');
            end

            T    = dload(fullfile(localDir, 'reginfo.tsv'));
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
    
                % rename contrast images and spmT images
                conName = {'con','spmT'};
                for n = 1:numel(conName)
                    oldName = fullfile(localDir, sprintf('%s_%2.4d.nii',conName{n},cname_idx));
                    newName = fullfile(localDir, sprintf('%s_%s.nii',conName{n},SPM.xCon(cname_idx).name));
                    movefile(oldName, newName);
                end % conditions (n, conName: con and spmT)

            end

            cd(currentDir);                                     
            
       case 'GLM:within_participant'

           currentDir = pwd;
            
            spm_get_defaults('cmdline', true);  % Suppress GUI prompts, no request for overwirte 
            
            if exist(localDir,"dir"); rmdir(localDir,'s'); end

            if strcmp(fit_to_day, 'common')

                if ~exist(subj_est_dir,"dir")
                    mkdir(subj_est_dir);
                end
                
                % Check for and delete existing SPM.mat file
                spm_file = fullfile(subj_est_dir, 'SPM.mat');
                if exist(spm_file, 'file')
                    delete(spm_file);
                end

                if isfile(fullfile(baseDir, sprintf('glm%d', glm),'hrf_params.tsv'))
                    P = dload(fullfile(baseDir, sprintf('glm%d', glm), 'hrf_params.tsv'));
                    hrf_params = P.P(P.sn==sn, :);
                else
                    hrf_params = [6 16 1 1 6 0 32];
                end

                disp('Using P:')
                disp(hrf_params)

                efcl_glm('GLM:make_event', 'sn', sn, 'glm', glm, 'day', day)
                efcl_glm('GLM:design', 'sn', sn, 'glm', glm, 'hrf_params', hrf_params, 'day', day, 'derivs', derivs)
                efcl_glm('GLM:estimate', 'sn', sn, 'glm', glm, 'day', day)
                efcl_glm('GLM:T_contrasts', 'sn', sn, 'glm', glm, 'day', day)

                S = load(fullfile(localDir,'SPM.mat'));
                S.SPM.swd = subj_est_dir;
                save(fullfile(localDir,'SPM.mat'), '-struct', 'S', 'SPM', spm_get_defaults('mat.format'));

                cd(currentDir);                                     % leave localDir before deleting it
                copyfile(fullfile(localDir,'*'), subj_est_dir);
                rmdir(localDir,'s');

                efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'con', 'day', day)      

            elseif strcmp(fit_to_day, 'separate')

                for i = 1:length(day)
                    subj_est_dir = fullfile(subj_est_dir, sprintf('day%d', day(i)));
                    if ~exist(subj_est_dir,"dir")
                        mkdir(subj_est_dir);
                    end
                    
                    % Check for and delete existing SPM.mat file
                    spm_file = fullfile(baseDir, subj_est_dir, 'SPM.mat');
                    if exist(spm_file, 'file')
                        delete(spm_file);
                    end
                    
                    if isfile(fullfile(baseDir, subj_est_dir, 'hrf_params.tsv'))
                        P = dload(fullfile(baseDir, subj_est_dir, 'hrf_params.tsv'));
                        hrf_params = P.P(P.sn==sn, :);
                    else
                        hrf_params = [6 16 1 1 6 0 32];
                    end
        
                    disp('Using P:')
                    disp(hrf_params)

                    efcl_glm('GLM:make_event', 'sn', sn, 'glm', glm, 'day', day(i))
                    efcl_glm('GLM:design', 'sn', sn, 'glm', glm, 'hrf_params', hrf_params, 'day', day(i), 'derivs', derivs)
                    efcl_glm('GLM:estimate', 'sn', sn, 'glm', glm, 'day', day(i))
                    efcl_glm('GLM:T_contrasts', 'sn', sn, 'glm', glm, 'day', day(i))
                    efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'con', 'day', day(i))
                end
            end

       case 'GLM:across_participants'
           
            % sn = [];
            % glm = [];
            % derivs = [0, 0];
            % day = [3, 9, 23];
            % baseline = 'common';
            % vararginoptions(varargin,{'sn', 'glm', 'day','derivs', 'baseline'})
            
            for s=sn
               efcl_glm('GLM:within_participant', 'sn', s, 'glm', glm, 'day', day, 'fit_to_day', fit_to_day)
            end
            
       case 'SURF:vol2surf'
            
            currentDir = pwd;

            res  = 32;          % resolution of the atlas. options are: 32, 164
            
            V = {};
            cols = {};
            if strcmp(type, 'spmT')
%                 filename = ['spmT_' id '.func.gii'];
                files = dir(fullfile(subj_est_dir, 'spmT_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'beta')
                SPM = load(fullfile(subj_est_dir,  'SPM.mat')); SPM=SPM.SPM;
                files = dir(fullfile(subj_est_dir, 'beta_*.nii'));
                files = files(SPM.xX.iC);
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'psc')
                files = dir(fullfile(subj_est_dir, 'psc_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'con')
                files = dir(fullfile(subj_est_dir, 'con_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'res')
                V{1} = fullfile(subj_est_dir, 'ResMS.nii');
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
    
            save(GL, fullfile(baseDir, wbDir, subj_id, ['glm' num2str(glm) '.'   type '.L.func.gii']))
    
            GR = surf_vol2surf(c1R,c2R,V,'anatomicalStruct','CortexRight', 'exclude_thres', 0.9, 'faces', hemRpial.faces);
            GR = surf_makeFuncGifti(GR.cdata,'anatomicalStruct', 'CortexRight', 'columnNames', cols);

            save(GR, fullfile(baseDir, wbDir, subj_id, ['glm' num2str(glm)  '.'  type '.R.func.gii']))
            
            cd(currentDir)
        
           
    end

end
