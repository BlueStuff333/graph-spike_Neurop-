function [idx] = calcIdx(M,K)

idx = zeros(K,M^K);
for i=0:K-1
    idx0 = zeros(M^(i+1),1);
    for j=1:M
        idx0((j-1)*M^i+1:j*M^i) = ones(M^i,1)*j;
    end
    idx(i+1,:) = repmat(idx0,M^(K-1-i),1);
end