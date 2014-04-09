function [A, Ao, evals, P, Po, Pbest] = ...
    mo_pso(generations, problem_function, n, max_params, min_params, num_obj, ...
    problem_func_params, p_mut, mut_w, swarm_size, inertia, c1, c2, chi)

% [Archive,Archive_objectives,samples, samples_objectives] = ...
%    mo_pso(generations, problem_function, n, max_param, min_param, 
%    num_obj, problem_func_params, Q, swarm_size)
%
%
% Implements the multi-objective particle swarm optimiser described in
% Fieldsend JE, Singh S. (2002) "A multi-objective algorithm based upon 
% particle swarm optimisation, an efficient data structure and turbulence"
% UK Workshop on Computational Intelligence (UKCI'02), 
% Birmingham, Uk, 2nd - 4th Sep 2002
%
% However due to the implementation language here, the effect of the data
% structure is simply emulated.
%  
% Assumes that all objectives are to be minimised
%
% inputs:
% 
% generations = number of iterations of algorithm
% problem_function = string containing the name of the objective
%   function to optimise, must take as arguments the decision vector
%   followed by the number of objectives and a structure of problem-specific
%   meta-parameters. The functions should return an array (1 by
%   D) of the D objectives evaluated
% n = number of decision parameters
% min_params = minimum value of decision parameters permitted (1 by n vector)
% max_params = maximum value of decision parameters permitted (1 by n
%   vector)
% num_obj = number of objectives
% problem_func_param = structure of problem-specific meta-parameters
% p_mut = probability of mutation (option, default 0.2)
% mut_w = standard deviation of guassian mutation (option, default 0.1)
% swarm_size = number of members in search population (option, default 20)
% inertia = inertia term (option, default 0.4)
% c1 = cognative constraint term (option, default 1.0)
% c2 = social constraint term (option, default 1.0)
% chi = overall constraint term (option, default 1.0)
%
% returns:
%
% Archive = matrix of archive decision vectors
% Archive_objectives = matrix of archive member evaluations
% evals = total number of function evaluations
%
% (c) 2014 Jonathan Fieldsend, University of Exeter


if ~exist('swarm_size','var')
    swarm_size = 20;
end
if ~exist('mut_w','var')
    mut_w = 0.1;
end
if ~exist('p_mut','var')
    p_mut = 0.2;
end
if ~exist('inertia','var')
    inertia = 0.4;
end
if ~exist('c1','var')
    c1 = 1.0;
end
if ~exist('c2','var')
    c2 = 1.0; 
end
if ~exist('chi','var')
    chi = 1.0;
end
range = max_params - min_params;
max_matrix = repmat(max_params,swarm_size,1);
min_matrix = repmat(min_params,swarm_size,1);

% INITIALISATION 

% initialise particles and Pbests
P = rand(swarm_size,n);
P = P.*repmat(range,swarm_size,1) + ...
    repmat(min_params,swarm_size,1);
Po=zeros(swarm_size,num_obj);
for i=1:swarm_size
    Po(i,:) = feval(problem_function,P(i,:),...
        num_obj,problem_func_params);
    Pbest(i).X = P(i,:);
    Pbest(i).Xo = Po(i,:);
end
evals = swarm_size;

% initialise velocities

V = rand(swarm_size,n);
V = V.*repmat(range,swarm_size,1) + ...
    repmat(min_params,swarm_size,1);
V = 2*(V - 0.5); % want random velocities of magnitude in both directions

% initialise_archive
[A, Ao] = update_Pareto_set([], [], P, Po);

for i=1:generations
    % get global and personal bests for this generation
    [G] = get_global_bests(A,Ao,P,Po,num_obj);
    [Pb] = get_personal_bests(Pbest);
    
    % update velocities 
    
    V = inertia*V + c1*rand(swarm_size,n).*(Pb-P) + ... % cognative
        c2*rand(swarm_size,n).*(G-P) + ... % social
        mut_w*randn(swarm_size,n).*repmat(range, swarm_size,1).*(rand(swarm_size,n)<p_mut); % turbulence
    
    % accelerate swarm members
    
    P = P + chi*V;
    
    % ensure legality
    
    P(P>max_matrix) = max_matrix(P>max_matrix);
    P(P<min_matrix) = min_matrix(P<min_matrix);
    
    % evaluate
    for j=1:swarm_size
        Po(j,:) = feval(problem_function,P(j,:),...
            num_obj,problem_func_params);
    end
    evals = evals + swarm_size;
    
    % update non-dominated global store
    [A, Ao] = update_Pareto_set(A, Ao, P, Po);
    
    % update non-dominated personal store
    for j=1:swarm_size
        [Pbest(j).X, Pbest(j).Xo] = update_Pareto_set(Pbest(j).X, Pbest(j).Xo, P(j,:), Po(j,:));
    end
    
    fprintf('Generation: %d, evaluations: %d, archive size %d\n', i, evals, size(Ao,1));
end


%----------------------------------------------
function [G, C, Co] = get_global_bests(A,Ao,P,Po,num_obj)

% preallocate composite point matrices for efficiency

C = zeros(ceil(size(A,1)/num_obj),size(Ao,2));
Co = zeros(ceil(size(A,1)/num_obj),size(Ao,2));
G = zeros(size(P));
point = 1;

% first construct composite points from A
while ( point<=ceil(size(A,1)/num_obj) ) % while there are still composite points to make
    if point == ceil(size(A,1)/num_obj)
        % making last composite point -- there may be few archived points
        % remaining than objective dimensions.
        for i=1:num_obj; % for each objective
            [~,I] = max(Ao(:,i));
            temp = Ao(I(1),:);
            temp_index = I(1);
            C(point,i) = I(1);
            Co(point,i) = Ao(I(1),i);
            Ao(I(1),:) = -inf; % ensure that it will not be considered again as a max
            if (Co(point,i) == -inf)
               % have reached a point where all archive members have been
               % used up, so use last_temp to fill in this and remaining
               % values -- can do this as will never occur when i==1 
               C(point,i-1:num_obj) = last_temp_index;
               Co(point,i-1:num_obj) = last_temp(1, i-1:num_obj);
               break;
            end
            last_temp = temp;
            last_temp_index = temp_index;
        end
    else % standard situation
        for i=1:num_obj; % for each objective
            [v,I] = max(Ao(:,i));
            C(point,i) = I(1);
            Co(point,i) = Ao(I(1),i);
            Ao(I(1),:) = -inf; % ensure that it will not be considered again as a max
            if (v==-inf)
               error('Have looped too many times in composite point construction'); 
            end
        end
    end
    point = point + 1;
end

% now we have the composite points, can get the global guides

for i=1:size(P,1)
    domed_indices = dominates_set(Po(i,:),Co);
    if isempty(domed_indices)
        first_non_domed = 1;
    else
        first_non_domed = domed_indices(end)+1;
        if (first_non_domed > size(Co,1))
            % corner case due to weak dominance 
            first_non_domed = size(Co,1);
        end
    end
    
    % now select as glocal guide first contributor to composite point which
    % weakly dominates the particle
    
    r = randperm(num_obj);
    gbest_set=0;
    for j=1:num_obj
        if sum(Ao(C(first_non_domed,r(j)),:) <= Po(i,:)) == num_obj
           G(i,:) = A(C(first_non_domed,r(j)),:);
           gbest_set=1;
           break;
        end
    end
    if (gbest_set==0)
       error('gbest not set'); 
    end
end

%----------------------------------------------
function [Pb] = get_personal_bests(Pbest)

% preallocate amtrices for efficiency
Pb = zeros(length(Pbest), size(Pbest(1).X,2));

% randomly select personal guide from each particle's non-dominated set
for i=1:length(Pbest)
    I = randperm(size(Pbest(i).X,1));
    Pb(i,:) = Pbest(i).X(I(1),:);
end

%----------------------------------------------
function [A, Ao] = update_Pareto_set(A, Ao, P, Po)

% given the non-dominated set A, and corresponding objectives in Ao, 
% updates with contents of P and Po

for i=1:size(P,1)
    I = set_dominates(Ao, Po(i,:));
    if isempty(I)==1; % if the ith is not dominated
        I = dominates_set(Po(i,:),Ao); % find dominated elite
        A(end+1,:) = P(i,:); % add to elite
        Ao(end+1,:) = Po(i,:);
        A(I,:)=[]; % remove dominated from elite
        Ao(I,:)=[];
    end
end

%----------------------------------------------
function indices= set_dominates(U,v)

% U = set of objective vectors, n by num_objectives
% v = single objective vector, 1 by num_objectives
%
% returns indices of elements of U which dominate v

if isempty(U);
    indices = [];
else
    Iwd=zeros(size(U,1),1);
    Id=zeros(size(U,1),1);
    for i=1:length(v)
        Iwd=Iwd+(U(:,i)<=v(i));
        Id=Id+(U(:,i)<v(i));
    end
    indices = find((Iwd==length(v))+(Id>0)==2);
end
%----------------------------------------------
function indices= dominates_set(v,U)

% U = set of objective vectors, n by num_objectives
% v = single objective vector, 1 by num_objectives
%
% returns indices of elements of U which are dominated by v

if isempty(U);
    indices = [];
else
    Iwd=zeros(size(U,1),1);
    Id=zeros(size(U,1),1);
    for i=1:length(v)
        Iwd=Iwd+(U(:,i)>=v(i));
        Id=Id+(U(:,i)>v(i));
    end
    
    indices = find((Iwd==length(v))+(Id>0)==2);
end
