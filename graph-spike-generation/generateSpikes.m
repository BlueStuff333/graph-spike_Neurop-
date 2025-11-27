function[firings, stl, y] = generateSpikes(adj, e_locs, i_locs)

Ne = sum(e_locs);
Ni = sum(i_locs);
n = Ne + Ni;

%% params for firing model
T = 10000; input_type = "lognormal";
input_strength_list = [10000];
noise_scalar_list = [0.6];
input_strength_scalar_E = 5; input_strength_scalar_I = 2;
num_rpt = 1;

% % interval of stimulus is generated from exponential distribution with mean of mu
% % the interval is further bounded within a certain range (40000, 70000)
num_input = 10; exp_mu = 5;
input_signal_t_interval = [exp_mu];
while length(input_signal_t_interval) < num_input
    tmp = exprnd(exp_mu);
    if (4 < tmp) && (tmp < 7)
        input_signal_t_interval = [input_signal_t_interval, tmp];
    end
end
input_signal_t = cumsum(input_signal_t_interval);
input_signal_t = floor(input_signal_t * T/num_input/exp_mu);

%% izhikevich model parameters
r = rand(n, 1);
re = r.*e_locs;
ri = r.*i_locs;
a = 0.02*e_locs + (0.02*i_locs+0.08*ri);
b = 0.2*e_locs + (0.25*i_locs-0.05*ri);
c = (-65*e_locs+15*re.^2) + -65*i_locs;
d = (8*e_locs-6*re.^2) + 2*i_locs;

for input_strength = input_strength_list
    for noise_scalar = noise_scalar_list
        
        input_noise_scalar_E = 5*noise_scalar;
        input_noise_scalar_I = 2*noise_scalar;
        
        for idx_rpt = 1:num_rpt
    
            %% input mask
            Ne_in = ceil(0.36*Ne); 
            Ni_in = ceil(0.36*Ni);
            [mask_E, mask_I] = InputMask_random(e_locs, i_locs,Ne_in, Ni_in);
            
            %% lognormal input
            pd = makedist('Lognormal','mu',7.5,'sigma',1);
        
            y = zeros(T,1);
            for i_t = 1:length(input_signal_t)
                t_start = input_signal_t(i_t);
                x_0 = (1:1:T-t_start)';
                y(t_start+1:T) = y(t_start+1:T) + pdf(pd,x_0)*input_strength;
            end
            
            
            %% start simulation
            % code source of Izhikevich model: https://www.izhikevich.org/publications/net.m
            
            v=-65*ones(n,1);
            u=b.*v;

            firings=zeros(T*(n), 2);
            i_firings = 1;

            % firing = [t,fired neuron idx;...]
            % I_t: thalamic input
            
            for t=1:T
                
                fired=find(v>=30); % indices of spikes
                n_fired = length(fired);
                firings(i_firings:i_firings+n_fired-1,:) = [t+0*fired, fired];
                i_firings = i_firings+n_fired;
                v(fired)=c(fired);
                u(fired)=u(fired)+d(fired);
                
                if input_type == "random"
                    % random thalamic input
                    thalamic = randn(n,1);
                    thalamic_i = 2 * (e_locs .* thalamic);
                    thalamic_e = 5 * (i_locs .* thalamic);
                    I = thalamic_i + thalamic_e;
                elseif input_type == "lognormal"
                    
                    % lognormal input
                    input_signal_to_E = input_strength_scalar_E * y(t);
                    input_signal_to_E = input_signal_to_E .* mask_E;
                    input_signal_to_I = input_strength_scalar_I * y(t);
                    input_signal_to_I = input_signal_to_I .* mask_I;
                    I_signal = input_signal_to_E + input_signal_to_I;
                    
                    % noise input
                    inputNoiseBase = randn(n, 1);
                    inputNoiseBase_E = e_locs .* inputNoiseBase;
                    inputNoiseBase_I = i_locs .* inputNoiseBase;
                    input_noise_to_E = [input_noise_scalar_E*inputNoiseBase_E]; % standard Gaussian noise
                    input_noise_to_I = [input_noise_scalar_I*inputNoiseBase_I];
                    I_noise = input_noise_to_E + input_noise_to_I;
                    
                    I = I_signal + I_noise;
                end

                I_t(:,t) = I;
                
                I=I+sum(adj(:,fired),2); % add the influence from spiked neuron
                
                v=v+0.5*(0.04*v.^2+5*v+140-u+I); % step 0.5 ms
                v=v+0.5*(0.04*v.^2+5*v+140-u+I); % for numerical
                u=u+a.*(b.*v-u); % stability
            end
            
            firings = firings(1:i_firings-1, :);
            
            %% analysis
            
            % convert firings to time series
            % spiking_time{i} is the spiking time series for the i-th neuron
            spiking_time = cell(n,1);
            for row = 1:size(firings,1)
                spiking_time{firings(row,2)} = [spiking_time{firings(row,2)}, firings(row,1)];
            end
            stl = zeros(n,1); % spiking time length
            for i = 1:n
                stl(i) = length(spiking_time{i});
            end
        end
    end
end