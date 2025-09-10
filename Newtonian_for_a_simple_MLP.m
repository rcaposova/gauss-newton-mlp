%% Newton's Algorithm for a Simple MLP with Sigmoid Activation

% Input-to-Hidden weights:
w_i1_h1 = -0.25;  
w_i1_h2 = 0;    
w_i1_h3 = 0.14;
w_i2_h1 = 0;      
w_i2_h2 = -0.17;  
w_i2_h3 = 0.16;
% Hidden-to-Output weights:
w_h1_o = 0.43;    
w_h2_o = 0.21;   
w_h3_o = -0.25;
% Input-to-Output (direct) weights:
w_i1_o = 0.33;    
w_i2_o = 0;

% Combine weights into a single vector (order: 
% 6 input-to-hidden, 3 hidden-to-output, 2 direct input-to-output)
w = [ w_i1_h1; w_i2_h1; % weights for hidden neuron 1
      w_i1_h2; w_i2_h2; % weights for hidden neuron 2
      w_i1_h3; w_i2_h3; % weights for hidden neuron 3
      w_h1_o; w_h2_o; w_h3_o; % hidden-to-output weights
      w_i1_o; w_i2_o ]; % direct input-to-output weights
nWeights = length(w);

% Network architecture
NInputs = 2;
NHidden = 3;
NOutputs = 1;
bias_hidden = zeros(NHidden,1); % biases for hidden neurons
bias_output = 0; % bias for the output neuron

% Define a dummy input and target (ASK WHY WE DID NOT GET THESE ASSIGNED?)
Inputs = [0.5; -0.5];  % 2x1 vector
target = 0.1;

%% Forward Pass with Sigmoid Activation
% Extract weight matrices from vector w:
w_in_hidden = reshape(w(1:6), [NInputs, NHidden]); % 2x3 matrix
w_hidden_out = w(7:9); % 3x1 vector
w_in_out = w(10:11); % 2x1 vector

% Compute net input to hidden neurons and apply sigmoid activation:
net_hidden = w_in_hidden' * Inputs + bias_hidden; 
hidden_output = 1 ./ (1 + exp(-net_hidden)); % Sigmoid: f(net)=1/(1+exp(-net))

% Compute network output by combining hidden-to-output and direct input-to-output paths:
out_hidden = w_hidden_out' * hidden_output;
out_direct = w_in_out' * Inputs;
output = out_hidden + out_direct + bias_output;

% Compute the error (t - y):
error = target - output;

%% Compute the Jacobian
% The Jacobian J (1 x nWeights) contains the partial_derivative_y/partial_derivative_w
J = zeros(1, nWeights);

% (a) derivatives with respect to input-to-hidden weights, w_{ij}^{(h)}:
for j = 1:NHidden
    % Compute the derivative of the sigmoid for hidden neuron j:
    d_hidden = hidden_output(j) * (1 - hidden_output(j));  
    for i = 1:NInputs
        idx = (j-1)*NInputs + i; % Index in the weight vector
        % According to Haykin:
        % partial_derivative_y/partial_derivative_w_{ij}^{(h)} = w_j^(o) * f'(net_j) * x_i
        J(idx) = w_hidden_out(j) * d_hidden * Inputs(i);
    end
end

% (b) derivatives with respect to hidden-to-output weights, w_j^(o):
for j = 1:NHidden
    idx = 6 + j; % because positions 7-9 in the weight vector
    % partial_derivative_y/partial_derivative_w_j^(o) = f(net_j)
    J(idx) = hidden_output(j);
end

% (c) derivatives with respect to direct input-to-output weights, w_i^(d):
for i = 1:NInputs
    idx = 9 + i; % because positions 10-11 in the weight vector
    % partial_derivative_y/partial_derivative_w_i^(d) = x_i
    J(idx) = Inputs(i);
end

%% Compute the Gradient and the Hessian Approximation
% Cost function: E = 0.5*(t - y)^2.
% Its gradient is: nablaE = -J^T (t - y) - move this section after the
% 107!!!
gradient = -J' * error;

% Hessian (Gauss–Newton approximation): H ≈ J^T J - divide by n inputs!!!
H = J' * J; 

% Apply damping for numerical stability (Levenberg–Marquardt modification):
% I decided to do this to ensure that the Newton update remains stable 
% and effective, particularly because the Hessian in this case is close to singular
lambda = 1e-4;
H_damped = H + lambda * eye(nWeights);

%% Compute the Newton (Gauss–Newton) Weight Update
% deltaw = -H^{-1} nablaE = H^{-1} J^T (t - y)
% since gradient = -J^T (t-y), we can also write:
% deltaw = -pinv(H_damped) * gradient)
delta = -pinv(H_damped) * gradient;  
% Alternatively, since -gradient = J' * error, we can say:
% delta = pinv(H_damped) * (J' * error);

% Update the weights:
w_new = w + delta;

%% Display the updated weights
fprintf('Newton Update');
disp(w_new);
