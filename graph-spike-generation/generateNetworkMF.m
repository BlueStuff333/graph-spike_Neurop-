function [adj] = generateNetworkMF(PK,LK,LKcum,N,isDirected,isBinary)
% generating graph
% adj : adjacent matrix
% class : the row which each node falls into
% PK,LK,LKcum: model parameters. PK: linking probabilities. LK: attribute
% probabilities. LKcum: cumulative LK
% N: network size
% isDirected,isBinary: graph is directed/undirected; binary/weighted

adj = zeros(N,N);
class = zeros(N,1);

%% generating nodes
for i = 1:N
    tmpNode = rand;
    class(i) = find(tmpNode<=LKcum,1);
end

%% generating links
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
%% convert weighted to binary graph
if (isBinary)
    adj(adj>0)=1;
end