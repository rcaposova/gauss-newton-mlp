%% Sunspot Series Prediction: Newton's Algorithm vs. Backpropagation
% This script creates a fully connected network with 10 inputs and 5 hidden nodes.
% It compares two training methods:
%   1. Newton's algorithm using an approximate Gauss–Newton Hessian with adaptive LM damping.
%   2. Classical backpropagation (gradient descent).
% Performance is measured by the NMSE on a test set. 
% The predictions and the convergence curves are plotted.

clear; close all; clc;

%% Load and preprocess the Sunspot data
load sunspot.dat;           
data = sunspot(:,2); % Use the sunspot numbers from column 2

% Normalize the data (zero mean, unit variance)
data = (data - mean(data)) / std(data);

% Create input–target pairs
windowSize = 10;
N_total = length(data) - windowSize;
X = zeros(windowSize, N_total); 
y = zeros(N_total, 1);     

for i = 1:N_total
    X(:, i) = data(i : i+windowSize-1);
    y(i) = data(i+windowSize);
end

% Split the data into training and test sets (70% training, 30% test)
trainRatio = 0.7;
N_train = floor(trainRatio * N_total);
X_train = X(:, 1:N_train);
y_train = y(1:N_train);
X_test  = X(:, N_train+1:end);
y_test  = y(N_train+1:end);

%% Define network architecture
NInputs = 10;
NHidden = 5;
NOutputs = 1;
% Parameter count is calculated as:
% input to hidden: 10x5 = 50 parameters
% bias hidden: 5 parameters
% hidden to output: 5 parameters
% bias output: 1 parameter
% total = 50 + 5 + 5 + 1 = 61.
numParams = NInputs * NHidden + NHidden + NHidden * NOutputs + NOutputs;

%% Training using Newton's Algorithm (with adaptive LM damping)
maxEpochs_Newton = 200;
tol_Newton = 1e-4;
lambda = 1e-4; % Initial damping factor
lambda_inc = 10; % Increase factor if update fails
lambda_dec = 10; % Decrease factor if update is successful
alpha = 0.1; % Step scaling factor alpha

rng(1); % added for reproducibility
% Combined weight vector (starting small so that the algorithm converges
% more effectively)
w = 0.1 * randn(numParams, 1);

% Preallocate convergence history vector for Newton
cost_history_newton = zeros(maxEpochs_Newton, 1);

% Compute initial cost (for reporting)
cost_old = 0;
for s = 1:size(X_train,2)
    x_sample = X_train(:, s);
    % Extract parameters:
    w_in_hidden = reshape(w(1:NInputs*NHidden), [NInputs, NHidden]);
    bias_hidden = w(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
    w_hidden_out = w(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
    bias_output = w(end);
    
    net_hidden = w_in_hidden' * x_sample + bias_hidden;
    h = 1 ./ (1 + exp(-net_hidden));
    yhat = w_hidden_out' * h + bias_output;
    cost_old = cost_old + 0.5*(y_train(s)-yhat)^2;
end

fprintf('--- Training with Newton''s Algorithm (Adaptive LM) ---\n');
for epoch = 1:maxEpochs_Newton
    grad = zeros(numParams, 1);
    H = zeros(numParams, numParams);
    cost = 0;
    
    % Extract parameters once per epoch:
    w_in_hidden = reshape(w(1:NInputs*NHidden), [NInputs, NHidden]);
    bias_hidden = w(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
    w_hidden_out = w(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
    bias_output = w(end);
    
    % Loop over all training samples:
    for s = 1:size(X_train, 2)
        x_sample = X_train(:, s);
        target_val = y_train(s);
        
        % Forward pass:
        net_hidden = w_in_hidden' * x_sample + bias_hidden;
        h = 1 ./ (1 + exp(-net_hidden)); % sigmoid activation
        yhat = w_hidden_out' * h + bias_output;
        
        err = target_val - yhat;
        cost = cost + 0.5 * err^2;
        
        % Compute Jacobian for this sample:
        jac = zeros(1, numParams);
        % (a) Derivatives with respect to w_in_hidden:
        for j = 1:NHidden
            dh = h(j) * (1 - h(j)); % sigmoid derivative
            for i = 1:NInputs
                idx = (j-1)*NInputs + i;
                jac(idx) = w_hidden_out(j) * dh * x_sample(i);
            end
        end
        % (b) Derivatives with respect to bias_hidden:
        for j = 1:NHidden
            idx = NInputs*NHidden + j;
            jac(idx) = w_hidden_out(j) * h(j) * (1 - h(j));
        end
        % (c) Derivatives with respect to w_hidden_out:
        for j = 1:NHidden
            idx = NInputs*NHidden + NHidden + j;
            jac(idx) = h(j);
        end
        % (d) Derivative with respect to bias_output:
        jac(end) = 1;
        
        % Accumulate gradient and approximate Hessian (using Gauss–Newton):
        grad = grad + jac' * err;
        H = H + (jac' * jac);
    end
    
    % Average over training samples:
    grad = grad / size(X_train,2);
    H = H / size(X_train,2);
    
    % Adaptive damping:
    H_damped = H + lambda * eye(numParams);
    
    % Compute Newton update:
    delta = alpha * pinv(H_damped) * grad;
    
    % Candidate new weights:
    w_candidate = w + delta;
    
    % Compute new cost with the candidate weights:
    cost_new = 0;
    w_in_hidden = reshape(w_candidate(1:NInputs*NHidden), [NInputs, NHidden]);
    bias_hidden = w_candidate(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
    w_hidden_out = w_candidate(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
    bias_output = w_candidate(end);
    for s = 1:size(X_train,2)
        x_sample = X_train(:, s);
        net_hidden = w_in_hidden' * x_sample + bias_hidden;
        h = 1 ./ (1 + exp(-net_hidden));
        yhat = w_hidden_out' * h + bias_output;
        cost_new = cost_new + 0.5*(y_train(s)-yhat)^2;
    end
    cost_new = cost_new / size(X_train,2);
    
    % Compute current cost from current weights:
    cost_current = cost / size(X_train,2);
    
    % Adaptive damping update: - bayesian training chapter 9
    if cost_new < cost_current
        % Accept update: decreases damping factor
        w = w_candidate;
        lambda = lambda / lambda_dec;
        cost_history_newton(epoch) = cost_new;
        fprintf('Newton Epoch %d: Cost = %f, ||delta|| = %f, lambda = %e (accepted)\n', epoch, cost_new, norm(delta), lambda);
    else
        % Reject update: increases damping factor
        lambda = lambda * lambda_inc;
        cost_history_newton(epoch) = cost_current;
        fprintf('Newton Epoch %d: Cost = %f, ||delta|| = %f, lambda = %e (rejected update)\n', epoch, cost_current, norm(delta), lambda);
    end
    
    if norm(delta) < tol_Newton
        cost_history_newton = cost_history_newton(1:epoch);
        break;
    end
end

%% Evaluate the Newton model on the test set:
y_pred_newton = zeros(size(X_test, 2), 1);
w_in_hidden = reshape(w(1:NInputs*NHidden), [NInputs, NHidden]);
bias_hidden = w(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
w_hidden_out = w(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
bias_output = w(end);
for s = 1:size(X_test, 2)
    x_sample = X_test(:, s);
    net_hidden = w_in_hidden' * x_sample + bias_hidden;
    h = 1 ./ (1 + exp(-net_hidden));
    y_pred_newton(s) = w_hidden_out' * h + bias_output;
end

NMSE_newton = sum((y_test - y_pred_newton).^2) / sum((y_test - mean(y_test)).^2);
fprintf('Newton NMSE: %f\n', NMSE_newton);

%% Training using classical backpropagation
maxEpochs_BP = 200;
learning_rate_BP = 0.001;
rng(2); % Different seed for bp for reproducibility
w_bp = 0.1 * randn(numParams, 1); % Combined weight vector for bp

% Preallocate convergence history for bp:
cost_history_bp = zeros(maxEpochs_BP, 1);

fprintf('\n--- Training with Backpropagation ---\n');
for epoch = 1:maxEpochs_BP
    grad_bp = zeros(numParams, 1);
    cost_bp = 0;
    
    % Extract current bp parameters:
    % Input-to-hidden weights:
    w_in_hidden = reshape(w_bp(1:NInputs*NHidden), [NInputs, NHidden]);
    % Hidden layer biases:
    bias_hidden = w_bp(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
    % Hidden-to-output weights:
    w_hidden_out = w_bp(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
    % Output bias:
    bias_output = w_bp(end);
    
    % Loop over each training sample:
    for s = 1:size(X_train, 2)
        x_sample = X_train(:, s);
        target_val = y_train(s);
        
        %%% Forward Pass %%%
        % Hidden layer net input and activation:
        net_hidden = zeros(NHidden, 1);
        h = zeros(NHidden, 1);
        for j = 1:NHidden
            % Starting with bias for hidden unit j:
            net_hidden(j) = bias_hidden(j);
            % Sum over each input:
            for i = 1:NInputs
                net_hidden(j) = net_hidden(j) + w_in_hidden(i, j) * x_sample(i);
            end
            % Apply the sigmoid activation function:
            h(j) = 1 / (1 + exp(-net_hidden(j)));
        end
        
        % Compute output layer activation:
        yhat = bias_output;
        for j = 1:NHidden
            yhat = yhat + w_hidden_out(j) * h(j);
        end
        
        % Compute error and accumulate cost:
        err = target_val - yhat;
        cost_bp = cost_bp + 0.5 * err^2;
        
        %%% Backpropagation %%%
        % Output layer delta (the derivative of a linear output is 1)
        delta_o = err;
        
        % Hidden layer delta:
        delta_h = zeros(NHidden, 1);
        for j = 1:NHidden
            delta_h(j) = w_hidden_out(j) * delta_o * h(j) * (1 - h(j));
        end
        
        % Gradients for input-to-hidden weights:
        grad_w_in_hidden = zeros(NInputs, NHidden);
        for j = 1:NHidden
            for i = 1:NInputs
                grad_w_in_hidden(i, j) = delta_h(j) * x_sample(i);
            end
        end
        
        % Gradients for hidden biases:
        grad_bias_hidden = zeros(NHidden, 1);
        for j = 1:NHidden
            grad_bias_hidden(j) = delta_h(j);
        end
        
        % Gradients for hidden-to-output weights:
        grad_w_hidden_out = zeros(NHidden, 1);
        for j = 1:NHidden
            grad_w_hidden_out(j) = delta_o * h(j);
        end
        
        % Gradient for output bias:
        grad_bias_output = delta_o;
        
        % Combine gradients into a single vector:
        grad_sample = [grad_w_in_hidden(:); grad_bias_hidden; grad_w_hidden_out; grad_bias_output];
        grad_bp = grad_bp + grad_sample;
    end
    
    % Update the weight vector using the summed gradient:
    w_bp = w_bp + learning_rate_BP * grad_bp;
    
    % Average cost for the epoch:
    cost_history_bp(epoch) = cost_bp / size(X_train, 2);
    
    if mod(epoch, 20) == 0
        fprintf('BP Epoch %d: Cost = %f\n', epoch, cost_history_bp(epoch));
    end
end

%%% Evaluation on test set %%%
y_pred_bp = zeros(size(X_test, 2), 1);
% Extract final bp parameters:
w_in_hidden = reshape(w_bp(1:NInputs*NHidden), [NInputs, NHidden]);
bias_hidden = w_bp(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
w_hidden_out = w_bp(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
bias_output = w_bp(end);

for s = 1:size(X_test, 2)
    x_sample = X_test(:, s);
    
    % Forward pass for test sample:
    net_hidden = zeros(NHidden, 1);
    h = zeros(NHidden, 1);
    for j = 1:NHidden
        net_hidden(j) = bias_hidden(j);
        for i = 1:NInputs
            net_hidden(j) = net_hidden(j) + w_in_hidden(i, j) * x_sample(i);
        end
        h(j) = 1 / (1 + exp(-net_hidden(j)));
    end
    
    yhat = bias_output;
    for j = 1:NHidden
        yhat = yhat + w_hidden_out(j) * h(j);
    end
    
    y_pred_bp(s) = yhat;
end

NMSE_bp = sum((y_test - y_pred_bp).^2) / sum((y_test - mean(y_test)).^2);
fprintf('BP NMSE: %f\n', NMSE_bp);

%% Plot predictions and convergence
figure;
subplot(2,1,1);
plot(y_test, 'k', 'LineWidth', 2); hold on;
plot(y_pred_newton, 'r', 'LineWidth', 1.5);
plot(y_pred_bp, 'g', 'LineWidth', 1.5);
legend('Actual Sunspots', 'Newton Prediction', 'Backprop Prediction');
title('Sunspot Series Approximation Comparison');
xlabel('Time Index (Test Set)');
ylabel('Normalized Sunspot Number');
grid on;

subplot(2,1,2);
plot(1:length(cost_history_newton), cost_history_newton, 'r-o'); hold on;
plot(1:length(cost_history_bp), cost_history_bp, 'g-o');
legend('Newton Convergence', 'Backprop Convergence');
xlabel('Epoch');
ylabel('Average Cost');
title('Training Convergence Comparison');
grid on;

%% Evaluate the models on the TRAINING set

% Newton's Algorithm predictions on training set:
y_pred_newton_train = zeros(size(X_train, 2), 1);
w_in_hidden = reshape(w(1:NInputs*NHidden), [NInputs, NHidden]);
bias_hidden = w(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
w_hidden_out = w(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
bias_output = w(end);
for s = 1:size(X_train, 2)
    x_sample = X_train(:, s);
    net_hidden = w_in_hidden' * x_sample + bias_hidden;
    h = 1 ./ (1 + exp(-net_hidden));
    y_pred_newton_train(s) = w_hidden_out' * h + bias_output;
end

% Backpropagation predictions on training set:
y_pred_bp_train = zeros(size(X_train, 2), 1);
w_in_hidden_bp = reshape(w_bp(1:NInputs*NHidden), [NInputs, NHidden]);
bias_hidden_bp = w_bp(NInputs*NHidden+1 : NInputs*NHidden+NHidden);
w_hidden_out_bp = w_bp(NInputs*NHidden+NHidden+1 : NInputs*NHidden+NHidden+NHidden);
bias_output_bp = w_bp(end);
for s = 1:size(X_train, 2)
    x_sample = X_train(:, s);
    net_hidden = zeros(NHidden, 1);
    h = zeros(NHidden, 1);
    for j = 1:NHidden
        net_hidden(j) = bias_hidden_bp(j);
        for i = 1:NInputs
            net_hidden(j) = net_hidden(j) + w_in_hidden_bp(i, j) * x_sample(i);
        end
        h(j) = 1 / (1 + exp(-net_hidden(j)));
    end
    y_pred_bp_train(s) = bias_output_bp + w_hidden_out_bp' * h;
end

%% Plot training set predictions
figure;
plot(y_train, 'k', 'LineWidth', 2); hold on;
plot(y_pred_newton_train, 'r', 'LineWidth', 1.5);
plot(y_pred_bp_train, 'g', 'LineWidth', 1.5);
legend('Actual Training Data', 'Newton Prediction', 'Backprop Prediction');
title('Training Set Approximation Comparison');
xlabel('Time Index (Training Set)');
ylabel('Normalized Sunspot Number');
grid on;
