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
    atlas = 'ROI';
    hrf_params = [5 12 1 1 6 0 32];
    derivs = [0, 0];
    vararginoptions(varargin,{'sn', 'day', 'type', 'glm', 'hrf_params', 'atlas','derivs'})

    glmEstDir = 'glm';
    behavDir = 'behavioural';
    % anatomicalDir = 'anatomicals';
    imagingDir = 'imaging_data';
    wbDir = 'surfaceWB';
    regDir = 'ROI';

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
                D.(fname) = [D.(fname); D_tmp.(fname)];
            end
        end
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
        day_id = sprintf('day%d', day);
        runs = spmj_dotstr2array(subj_row.(sprintf('FuncRuns_day%d', day)){1});
    else
        runs = [];
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
                    events.Onset = [events.Onset; D.startTimeReal(D.chordID == chordID & D.day == day(d)) + 500];
                    events.Duration = [events.Duration; D.execMaxTime(D.chordID == chordID & D.day == day(d))];
                    events.chordID = [events.chordID; repmat({sprintf('%d', chordID)}, [60, 1])];
                    events.day = [events.day; repmat({sprintf('%d', day(d))}, [60, 1])];
                    events.eventtype = [events.eventtype; repmat({sprintf('%d,day%d', chordID, day(d))}, [60, 1])];
                end
            end
            
            events = struct2table(events);
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            varargout{1} = events;
            
        case 'GLM:make_event'
            
            operation  = sprintf('GLM:make_glm%d', glm);
            
            events = efcl_glm(operation, 'sn', sn, 'day', day);
            events = events(ismember(events.BN, runs), :);
            
            %% export
            output_folder = fullfile(baseDir, behavDir, 'day1', subj_id);
            writetable(events, fullfile(output_folder, sprintf('glm%d_events.tsv', glm)), 'FileType', 'text', 'Delimiter','\t')

%             if ~isfolder(fullfile(baseDir, [glmEstDir num2str(glm)] , subj_id))
%                 mkdir(fullfile(baseDir, [glmEstDir num2str(glm)], subj_id))
%             end
            
        case 'GLM:design'

            % Import globals from spm_defaults 
            global defaults;
            if (isempty(defaults)) 
                spm_defaults;
            end 
            
            currentDir = pwd;

            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end

            chords = unique(D.chordID);

            % Load data once, outside of session loop
            % D = dload(fullfile(baseDir,behavDir,subj_id, sprintf('smp2_%d.dat', sn)));
            events_file = sprintf('glm%d_events.tsv', glm);

            Dd = dload(fullfile(baseDir,behavDir, 'day1', subj_id, events_file));
            
            regressors = unique(Dd.eventtype);
            nRegr = length(regressors); 

            % init J
            J = [];
            T = [];
            J.dir = {fullfile(baseDir,sprintf('glm%d', glm), subj_id)};
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
                J.sess(run).scans = {fullfile(baseDir, imagingDir, subj_id, sprintf('u%s_run_%02d.nii', subj_id, run))};
        
                % Preallocate memory for conditions
                J.sess(run).cond = repmat(struct('name', '', 'onset', [], 'duration', []), nRegr/length(day), 1);

                regr = 1;            
                for d = 1:length(day)
                    for chordID = chords'

                        rows = find(Dd.BN == run & Dd.day == day(d) & Dd.chordID == chordID);
    
                        if ~isempty(rows)
     
                            % Regressor name
                            J.sess(run).cond(regr).name = sprintf('%d,%d', day(d), chordID);
                            
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
                            TT.run       = mod(run - 1, 10) + 1;
                            TT.name      = sprintf('%02d,%d', day(d), chordID);      
        
                            T = addstruct(T, TT);

                            regr = regr + 1;
                    
                        end
                    end
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
            subj_est_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id);                
            SPM = load(fullfile(subj_est_dir,'SPM.mat'));
            SPM.SPM.swd = subj_est_dir;
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
            glm_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id); 

            % load the SPM.mat file
            SPM = load(fullfile(glm_dir, 'SPM.mat')); SPM=SPM.SPM;

            if replace_xCon
                SPM  = rmfield(SPM,'xCon');
            end

            T    = dload(fullfile(glm_dir, 'reginfo.tsv'));
            T.name = cellstr(string(T.name));
            T.name = cellfun(@(a,b) [a ',' b], T.name(:,1), T.name(:,2), 'UniformOutput', false);
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
                % save('SPM.mat', 'SPM','-v7.3');
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
            efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'beta', 'day', day)
            efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'res', 'day', day)
            efcl_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'con', 'day', day)
             % efcl_glm('HRF:ROI_hrf_get', 'sn', sn, 'glm', glm, 'hrf_params', hrf_params, 'day', day)
            
       case 'SURF:vol2surf'
            
            currentDir = pwd;

            res  = 32;          % resolution of the atlas. options are: 32, 164

            glmEstDir = [glmEstDir num2str(glm)];
            
            V = {};
            cols = {};
            if strcmp(type, 'spmT')
%                 filename = ['spmT_' id '.func.gii'];
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'spmT_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'beta')
                SPM = load(fullfile(baseDir, glmEstDir, subj_id, 'SPM.mat')); SPM=SPM.SPM;
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'beta_*.nii'));
                files = files(SPM.xX.iC);
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'psc')
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'psc_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'con')
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'con_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'res')
                V{1} = fullfile(baseDir, glmEstDir, subj_id, 'ResMS.nii');
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
    
            save(GL, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.'   type '.L.func.gii']))
    
            GR = surf_vol2surf(c1R,c2R,V,'anatomicalStruct','CortexRight', 'exclude_thres', 0.9, 'faces', hemRpial.faces);
            GR = surf_makeFuncGifti(GR.cdata,'anatomicalStruct', 'CortexRight', 'columnNames', cols);

            save(GR, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.'  type '.R.func.gii']))
            
            cd(currentDir)
            
        case 'GLM:hrf'
            
            SPM = load(fullfile(baseDir, [glmEstDir num2str(glm)], subj_id, 'SPM.mat'));
            
            Hem = {'L', 'R'};
            rois = {'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'};
            R = {};
            r = 1;
            for h = 1:length(Hem)
                for rr = 1:length(rois)
                    R{r}.hem = Hem{h};
                    R{r}.name = rois{rr};
                    R{r}.file = fullfile(baseDir, 'ROI', subj_id, sprintf('%s.%s.%s.nii', atlas, Hem{h}, rois{rr}));
                    R{r}.value = 1;
                    R{r}.type = 'image';
                    R{r}.threshold = .5;
                    r = r+1;
                end
            end

            R = region_calcregions(R);
            
            [y_raw,y_adj,y_hat,y_res, B, y_filt] = region_getts(SPM, R);
            
            TR = 1;
            nScan = 336;
            
            Dd = []; T = [];
            Dd.ons = (D.startTimeReal(1:2:end) / 1000) / TR;
            Dd.ons = Dd.ons + (D.BN(1:2:end) - 1) * nScan;
            Dd.block = D.BN(1:2:end);   
            Dd.chordID = D.chordID(1:2:end);
            Dd.day = D.day(1:2:end);
            pre = 6;
            post= 18;
            for r=1:size(y_raw,2)
                for i=1:size(Dd.block,1)
                    Dd.y_adj(i,:)=cut(y_adj(:,r),pre,round(Dd.ons(i)),post,'padding','nan')';
                    Dd.y_hat(i,:)=cut(y_hat(:,r),pre,round(Dd.ons(i)),post,'padding','nan')';
                    Dd.y_res(i,:)=cut(y_res(:,r),pre,round(Dd.ons(i)),post,'padding','nan')';
                    Dd.y_raw(i,:)=cut(y_raw(:,r),pre,round(Dd.ons(i)),post,'padding','nan')';
                end
                
                % Add the event and region information to tje structure. 
                len = size(Dd.ons,1);                
                Dd.SN        = ones(len,1)*sn;
                Dd.region    = ones(len,1)*r;
                Dd.name      = repmat({R{r}.name},len,1);
                Dd.hem       = repmat({R{r}.hem},len,1);
                T           = addstruct(T,Dd);
            end
            
            save(fullfile(baseDir, [glmEstDir num2str(glm)], subj_id, 'T.mat'), 'T','-v7');
        
            
    end

end