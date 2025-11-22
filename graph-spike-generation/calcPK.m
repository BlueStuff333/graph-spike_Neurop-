function [PK] = calcPK(M,K,idx,P)
% calculate K-1 th kronecker product of P
PK = zeros(M^K,M^K);
for i=1:M^K
    for j=1:M^K
            PK(i,j) = prod(diag(P(idx(:,i),idx(:,j))));
    end
end