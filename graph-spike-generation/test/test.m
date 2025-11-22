clc;

data = load("net_isW_1_32_amp_EE_70_EI_270_IE_280_II_1080.mat");
S = data.S;

% disp(S)

% disp(sum(S>0, 'all'));
% disp(sum(S<0, 'all'));

absS = abs(S);
fprintf('Connections per neuron: %.1f\n', nnz(absS)/n);
fprintf('Density: %.4f\n', nnz(absS)/n^2);
histogram(sum(S>0, 2)); title('In-degree distribution');

% disp(mean(S(S>0)));
% disp(mean(S(S<0)));
% disp(mean(S(S~=0)));

% G = digraph(S);
% plot(G)