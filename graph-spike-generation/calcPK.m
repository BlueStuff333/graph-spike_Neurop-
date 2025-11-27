function [PK] = calcPK(M,K,P)
% calculate K-1 th kronecker product of P
PK = zeros(M^K,M^K);
% for i=1:M^K
%     for j=1:M^K
%             PK(i,j) = prod(diag(P(idx(:,i),idx(:,j))));
%     end
% end
nd = ndims(P);

if nd == 2
    % ----- Unweighted / single-resolution case -----
    PK = P;
    for k = 2:K
        PK = kron(PK, P);
    end
    % size(PK) = [M^K, M^K]

elseif nd == 3
    % ----- Weighted case: P(:,:,r) per weight level r -----
    R  = size(P, 3);
    MK = M^K;
    PK = zeros(MK, MK, R);

    for r = 1:R
        PR = P(:,:,r);   % MxM
        PKr = PR;
        for k = 2:K
            PKr = kron(PKr, PR);
        end
        PK(:,:,r) = PKr;
    end

else
    error('P must be 2-D (MxM) or 3-D (MxMxR).');
end