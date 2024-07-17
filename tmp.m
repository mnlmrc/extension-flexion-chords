% function [v_g, v_gs, v_gse] = reliability_var(Y, subj_vec, part_vec, varargin)
% alighavam79@gmail.com
%
% Description:
%       variance decomposition on dataset. Say the multi-channel data
%       y_ij follow the following format:
%               y_ij = g + s_i + e_ij
%       i, being the subject number and j, the partition/session number.
%
%       Assuming a) g, s_i, e_ij are mutually independent b) e_ij and s_i
%       are i.i.d, we can estimate the term variances as follows:
%       
%       Across subjects:
%       v_g = E[y_ij, y_kl]
%       Within subject, Across run:
%       v_g + v_s = E[y_ij, y_ik]
%       Within observation/partition:
%       v_g + v_s + v_e = E[y_ij, y_ij]
%
%       To develop estimators for these quantities we replace the 
%       Expectation with the mean over all possible pairings.
%
% INPUT:
%       Y: Data vector/matrix vertically concatenated for subjects and
%       partitions. In case of using dataframe this would be C.data.
%
%       subj_vec: column vector of subject numbers. In case of using
%       dataframe this would be C.sn
%
%       part_vec: column vector of partition numbers. Partitions could be
%       sessions/blocks/runs within a subject. In case of using dataframe
%       this would be something like C.sess/C.partition/...
%
%       varargin: 
%       'cond_vec': column vector of condition numbers. If a cond_vec is
%       inputted, the reliability is calculated separately for different
%       conditions. The output v_g, v_gs, v_gse will be in form of a cell
%       array containing the estimations for different conditions in each
%       cell.
%
%       'centered': 1 or 0. If 1, centers the data within observation y_ij 
%       before variance decomposition.
%
% OUTPUT:
%       v_g: estimated across subject variance (global effect).
%       v_gs: estimated (Within subject, Across run) variance.
%       v_gse: estimated (Within observation) variance.
%
%
% HOW TO USE: 
% % Here is an example of how to use the function.
% % Experiment: Play with the value of n to see how it effects the estimation
% % of the variance portions.
%
% % number of data points in each partition:
% n = 30;
% 
% % global effect:
% g = normrnd(0, 2, n, 1);   
% % subject 1 effect:
% s1 = normrnd(0, 1, n, 1);  
% % subejct 2 effect:
% s2 = normrnd(0, 1, n, 1);  
% 
% % subject 1 data:
% x1 = [g + s1 + normrnd(0,0.3,n,1) ; g + s1 + normrnd(0,0.3,n,1)];
% % subject 2 data:
% x2 = [g + s2 + normrnd(0,0.3,n,1) ; g + s2 + normrnd(0,0.3,n,1)];
% 
% Y = [x1 ; x2];
% subj_vec = kron([1;2],ones(2*n,1));
% part_vec = kron([1;2;1;2],ones(n,1));
% 
% [v_g, v_gs, v_gse] = reliability_var(Y, subj_vec, part_vec);
% 
% % theoretical var decomp:
% fprintf('Theoretical:\nvar_g = %.4f , var_s = %.4f , var_e = %.4f\n',4/(4+1+0.09),1/(4+1+0.09),0.09/(4+1+0.09))
% 
% % estimated:
% fprintf('Estimated:\nvar_g = %.4f , var_s = %.4f , var_e = %.4f\n\n',v_g/v_gse,(v_gs-v_g)/v_gse,(v_gse-v_gs)/v_gse)

% Parameters for the simulation
numSubjects = 5;    % Number of subjects
numConditions = 3;  % Number of conditions
numPartitions = 4;  % Number of partitions per condition
numDataPoints = 100; % Number of data points per partition

%% Initialize vectors
Y = [];
subj_vec = [];
part_vec = [];
cond_vec = [];

% Simulate data for each subject, condition, and partition
for subj = 1:numSubjects
    for cond = 1:numConditions
        for part = 1:numPartitions
            % Generate random data for this subject, condition, and partition
            data = normrnd(0, 1, numDataPoints, 5);
            
            % Append data to Y
            Y = [Y; data];
            
            % Append corresponding subject, partition, and condition labels
            subj_vec = [subj_vec; subj * ones(numDataPoints, 1)];
            part_vec = [part_vec; part * ones(numDataPoints, 1)];
            cond_vec = [cond_vec; cond * ones(numDataPoints, 1)];
        end
    end
end

% Convert vectors to column vectors
subj_vec = subj_vec(:);
part_vec = part_vec(:);
cond_vec = cond_vec(:);

% check for nans:
if sum(isnan(Y),'all')~=0
    warning('Input data contains nan elements')
end


% % handling input arguments:
% cond_vec = ones(size(subj_vec));
% centered = 1;   % centers the data by default.
% vararginoptions(varargin,{'cond_vec','centered'})

subjects = unique(subj_vec);
partitions = unique(part_vec);
conds = unique(cond_vec);

%% estimating v_gse and v_gs:
v_gs = 0;
v_gse = 0;
if length(conds)>1
    v_gs = cell(length(conds),1);
    v_gse = cell(length(conds),1);
    for i = 1:length(conds)
        v_gs{i} = 0;
        v_gse{i} = 0;
    end
end

% container for subj data:
subj_data = {};

% loop on conditions:
for k = 1:length(conds)
    % loop on subjects:
    for i = 1:length(subjects)
        % matrix to hold subject data:
        A = [];
        % loop on partitions:
        for j = 1:length(partitions)
            % getting the data for subj i , partition j:
            part_data = Y(subj_vec==subjects(i) & part_vec==partitions(j) & cond_vec==conds(k),:);
            
            % center the data:
            % if centered
            %     % removing the mean of each channel of y:
            %     part_data = part_data - mean(part_data,1);
            % end
    
            % adding the partition data to A:
            A = [A, part_data(:)];
        end
        % storing subject data:
        subj_data{i,k} = A;
    
        % covariance-variance matrix of subject data:
        B = A' * A;
    
        % number of partitions:
        N = size(A,2);
        
        % sum of diagonal elems (y_ij' * y_ij):
        tmp_v_gse = trace(B)/N/length(subjects);

        % sum of off-diagonal elems (y_ij' * y_ik):
        mean_cov = B .* (1-eye(N));
        mean_cov = sum(mean_cov(:))/(N*(N-1));
        tmp_v_gs = mean_cov/length(subjects);

        if length(conds)>1
            v_gse{k} = v_gse{k} + tmp_v_gse;
            v_gs{k} = v_gs{k} + tmp_v_gs;
        else
            v_gse = v_gse + tmp_v_gse;
            v_gs = v_gs + tmp_v_gs;
        end
    end
end
    
% estimating v_g:
v_g = 0;
if length(conds)>1
    v_g = cell(length(conds),1);
    for i = 1:length(conds)
        v_g{i} = 0;
    end
end

N = length(subjects);
for k = 1:length(conds)
    for i = 1:N-1
        for j = i+1:N
            % across subjects covariance matrix (y_ij' * y_kl):
            B = subj_data{i,k}' * subj_data{j,k};
            tmp_v_g = sum(B(:))/size(B,1)^2/(N*(N-1)/2);

            if length(conds)>1
                v_g{k} = v_g{k} + tmp_v_g;
            else
                v_g = v_g + tmp_v_g;
            end
        end
    end
end