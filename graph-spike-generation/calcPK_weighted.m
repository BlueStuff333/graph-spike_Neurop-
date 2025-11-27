function PK = calcPK_weighted(M, K, P)
    % P: M x M x R
    % PK: (M^K) x (M^K) x R
    R = size(P, 3);
    MK = M^K;
    PK = zeros(MK, MK, R);
    for r = 1:R
        PK(:,:,r) = calcPK(M, K, P(:,:,r));
    end
end