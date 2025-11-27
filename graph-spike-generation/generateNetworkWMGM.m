function [adj] = generateNetworkWMGM(PK, LK, LKcum, N, isDirected, isBinary)
% generateNetworkMF  Generate (weighted) multifractal network.
%
%   PK : Q x Q          (MF)  linking parameter per class pair
%        Q x Q x R      (WMGM)  probability per weight level r=1..R
%   LK : Q x 1          attribute probabilities (for node classes)
%   LKcum : Q x 1       cumulative LK (for sampling classes)
%   N  : number of nodes
%   isDirected, isBinary : flags for directed/undirected, weighted/binary
%
%   adj   : N x N adjacency matrix
%   class : N x 1, class index of each node

adj = zeros(N,N);
class = zeros(N,1);

% determine size parameters
szPk = size(PK);
Q = szPk(1);
if numel(szPk) < 3
    R = 1; % single weight level (MF)
else
    R = szPk(3); % multiple weight levels (WMGM)
end

%% gen nodes
% for i = 1:N
%     tmpNode = rand;
%     class(i) = find(tmpNode <= LKcum, 1);
% end
u = rand(N, 1);         % N random uniforms
% LKcum is Q×1, we want thresholds: [0, LKcum(1), LKcum(2), ... ,1]
edges = [0; LKcum(:)];
[~, class] = histc(u, edges);

%% build cdf for edge weights

if R == 1
    % old behavior, single weight level
    rMax = 14;
    % w = zeros(1,rMax+1);
    w = 0:rMax;
    Q = length(LK);
    cmf = zeros(Q,Q,rMax+1);
    for q=1:Q
        for l = 1:Q
            p = PK(q,l);
            cmf(q,l,1) = 1-p;
            for r=1:rMax
                cmf(q,l,r+1) = cmf(q,l,r) + (1-p)*p^r;
            end
        end
    end

    if(isDirected)
        %% directed graph, unsymmetric P
        for i = 1:N
    %         for j=1:N
            for j = [1:i-1,i+1:N]
                tmpLink = rand;
                % %             p=PK(class(i),class(j));
                % %             if(tmpLink<p)
                % %             adj(i,j) = ceil(log(tmpLink/(1-p))/log(p));
                % %             adj(j,i) = adj(i,j);
                % %             end
                if(tmpLink>cmf(class(i),class(j),end))
                    adj(i,j) = w(rMax+1);
                else
                    adj(i,j) = w(find(tmpLink<=cmf(class(i),class(j),:),1));
                end
            end
        end
    else
        %% undirected graph, symmetric P
        for i = 1:N
            for j = i+1:N
                tmpLink = rand;
                % %             p=PK(class(i),class(j));
                % %             if(tmpLink<p)
                % %             adj(i,j) = ceil(log(tmpLink/(1-p))/log(p));
                % %             adj(j,i) = adj(i,j);
                % %             end
                if(tmpLink>cmf(class(i),class(j),end))
                    adj(i,j) = w(rMax+1);
                else
                    adj(i,j) = w(find(tmpLink<=cmf(class(i),class(j),:),1));
                end
                adj(j,i) = adj(i,j);
            end
        end
    end
else
    % new behavior, multiple weight levels
    % PK(q,l,r) is the probability of weight level r (before normalisation).
    % We also define P(weight=0) = 1 - sum_r PK(q,l,r), clipped at 0.
    % We then sample from the levels {0, w(1), ..., w(R)}.
    w = 1:R; 
    cmf = zeros(Q,Q,R+1); % cumulative mass function over weights 0..R

    for q = 1:Q
        for l = 1:Q
            p_vec = squeeze(PK(q,l,:));

            % normalize
            p_sum = sum(p_vec);
            if p_sum > 1
                p_vec = p_vec / p_sum;
                p_sum = 1;
            end

            p0 = max(0, 1 - p_sum);

            cmf(q,l,1) = p0;
            for r = 1:R
                cmf(q,l,r+1) = cmf(q,l,r) + p_vec(r);
            end
        end
    end

    if (isDirected)
        for i = 1:N
            for j = [1:i-1, i+1:N]
                tmpLink = rand;
                r_idx = find(tmpLink <= cmf(class(i), class(j),:), 1);

                if isempty(r_idx) || r_idx == 1
                    % r_idx == 0 -> weight 0, no edge, leave adj(i,j)=0
                    continue;
                else
                    adj(i,j) = w(r_idx - 1); % -1 because cmf index 1 corresponds to weight 0
                end
            end
        end
    else
        % undirected
        for i = 1:N
            for j = i+1:N
                tmpLink = rand;
                r_idx = find(tmpLink <= cmf(class(i), class(j),:), 1);

                if ~isempty(r_idx) && r_idx > 1
                    adj(i,j) = w(r_idx - 1);
                    adj(j,i) = adj(i,j);
                end
            end
        end
    end
    %% convert to weighted binary graph if requested
    if isBinary
        adj(adj > 0) = 1;
    end
end
    
