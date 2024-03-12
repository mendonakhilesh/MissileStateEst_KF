% Parameters
T = 1000; % Total time steps
p0 = 0.5; % Initial probability of being in state 1
lambda = 0.1; % Transition rate from state 1 to state -1
mu = 0.2; % Transition rate from state -1 to state 1

% Initialize signal and time vector
signal = zeros(1, T); % Initialize signal vector
time = 1:T; % Time vector

% Generate random telegraph signal
state = rand(1) < p0; % Initial state
for t = 1:T
    signal(t) = 100*(2*state - 1); % Assign current state to signal (1 maps to 1, 0 maps to -1)
    if state == 0
        state = rand(1) < lambda; % Transition to state 1 with probability lambda
    else
        state = rand(1) > mu; % Transition to state -1 with probability mu
    end
end

% Plot random telegraph signal
plot(time, signal);
xlabel('Time');
ylabel('Signal');
title('Random Telegraph Signal Model');
ylim([-120, 120]); % Adjust ylim for signal between -1 and 1

