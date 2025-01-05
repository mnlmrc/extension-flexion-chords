function efcl_glm(what, varargin)

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
    vararginoptions(varargin,{'sn', 'day'})

    glmEstDir = 'glm';
    behavDir = 'behavioural';

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

            D = dload(fullfile(baseDir, behavDir, day_id, subj_id, ['efcl_' sn '.dat']));

            

    end

end