% Global Configurations
display_plot = false;
display_plot = true;
single_run_plot = false;
single_run_plot = true;

number_of_simulations = 1000;

% Global Constants
V_C = 300.0; % [ft/sec]
T_F = 10.0; % [sec]
R1 = 15.0*(10.0^(-6)); % [rad^2 sec]
R2 = 1.67*(10.0^(-3)); % [rad^2 sec^3]
TAU = 2.0; % [sec]
W = 100.0^2; % [(ft/sec^2)^2]
DT = 0.01; % [sec]
T = 0:DT:9.99; % [sec]
A_MEAN = 0.0;
A_VAR = 100.0^2;
Y_MEAN = 0.0;
Y_VAR = 0.0^2;
V_MEAN = 0.0;
V_VAR = 200.0^2;

% Global Data Structures & Data
P_AVG_DS = zeros(3, 3, numel(T));
ERROR_DS = zeros(3, 1, numel(T), number_of_simulations);
TOTAL_ERROR = zeros(3, 1, numel(T));
RESIDUAL_DS = zeros(1, numel(T), number_of_simulations);

for jdx = 1:number_of_simulations
    disp(['Simulation #: ', num2str(jdx)]);
    
    % State Space
    F = [0.0, 1.0, 0.0; 0.0, 0.0, -1.0; 0.0, 0.0, -(1.0/TAU)];
    B = [0.0; 1.0; 0.0];
    G = [0.0; 0.0; 1.0];
    
    % Initial Values
    H_init = [1.0/(V_C*T_F), 0.0, 0.0];
    V_init = R1 + (R2/(T_F^2));
    y_init = 0.0;
    v_init = normrnd(V_MEAN, sqrt(V_VAR));
    at_init = normrnd(0.0, sqrt(A_VAR));
    
    N_MEAN = 0.0;
    N_VAR = V_init/DT;
    n = normrnd(N_MEAN, sqrt(N_VAR)); % Noise
    
    W_at = normrnd(A_MEAN, sqrt(A_VAR/DT));
    
    % Kalman Filter
    P_init = [0.0, 0.0, 0.0; 0.0, 200.0^2, 0.0; 0.0, 0.0, 100.0^2];
    P_history = zeros(3, 3, numel(T));
    K_history = zeros(3, 1, numel(T));
    Z_history = zeros(1, numel(T));
    x_hat_history = zeros(3, 1, numel(T));
    dx_hat_history = zeros(3, 1, numel(T));
    x_history = zeros(3, 1, numel(T));
    dx_history = zeros(3, 1, numel(T));
    err_history = zeros(3, 1, numel(T));
    residual = zeros(1, numel(T));
    
    P_history(:, :, 1) = P_init;
    K_history(:, :, 1) = (P_init * H_init') * (V_init^(-1));
    
    % Initialize actual states
    x_history(:, :, 1) = [y_init; v_init; at_init];
    dx_history(:, :, 1) = F * x_history(:, :, 1) + G * W_at;
    Z_history(1) = H_init * x_history(:, :, 1) + n;
    
    % Initialize estimate states
    x_hat_history(:, :, 1) = [0; 0; 0];
    dx_hat_history(:, :, 1) = F * x_hat_history(:, :, 1) + K_history(:, :, 1) * (Z_history(1) - H_init * x_hat_history(:, :, 1));
    err_history(:, :, 1) = [0; 0; 0];
    
    for idx = 1:(numel(T) - 1)
        H = [1.0/(V_C * (T_F - T(idx))), 0.0, 0.0];
        V = R1 + (R2/((T_F - T(idx))^2));
        
        % Update Variance
        P_dot = F * P_history(:, :, idx) + P_history(:, :, idx) * F' - ((P_history(:, :, idx) * H') * (V^(-1)) * H * P_history(:, :, idx)) + G * W * G';
        P_history(:, :, idx+1) = P_history(:, :, idx) + P_dot * DT;
        
        % Update Kalman Gain
        K_history(:, :, idx+1) = (P_history(:, :, idx) * H') * (V^(-1));
        
        % Update with new iteration of noise
        n = normrnd(N_MEAN, sqrt(V/DT));
        W_at = normrnd(A_MEAN, sqrt(A_VAR/DT));
        
        % Update actual states
        dx_history(:, :, idx+1) = F * x_history(:, :, idx) + G * W_at;
        x_history(:, :, idx+1) = x_history(:, :, idx) + dx_history(:, :, idx+1) * DT;
        Z_history(idx+1) = H * x_history(:, :, idx+1) + n;
        
        % Update estimate states and store in global data structure
        dx_hat_history(:, :, idx+1) = F * x_hat_history(:, :, idx) + K_history(:, :, idx+1) * (Z_history(idx+1) - H * x_hat_history(:, :, idx));
        x_hat_history(:, :, idx+1) = x_hat_history(:, :, idx) + dx_hat_history(:, :, idx+1) * DT;
        
        % Residual & Error
        residual(idx+1) = Z_history(idx+1) - (H * x_hat_history(:, :, idx+1));
        RESIDUAL_DS(:, idx+1, jdx) = residual(idx+1);
        err_history(:, :, idx+1) = x_hat_history(:, :, idx+1) - x_history(:, :, idx+1);
        ERROR_DS(:, :, idx+1, jdx) = err_history(:, :, idx+1);
    end
    
    TOTAL_ERROR = TOTAL_ERROR + ERROR_DS(:, :, :, jdx);
    
    % Plots
    if display_plot && single_run_plot
        figure(1);
        title('Filter Gain History');
        plot(T, squeeze(K_history(1, :, :))');
        hold on;
        plot(T, squeeze(K_history(2, :, :))', '--');
        plot(T, squeeze(K_history(3, :, :))', ':');
        ylabel('Kalman Filter Gains');
        xlabel('Time since launch [s]');
        hold off;
        legend('K1', 'K2', 'K3');
        grid on;
        pause(0.1);
        
        figure(2);
        title('Evolution of the Estimation Error RMS');
        plot(T, sqrt(squeeze(P_history(1, 1, :))));
        hold on;
        plot(T, sqrt(squeeze(P_history(2, 2, :))), '--');
        plot(T, sqrt(squeeze(P_history(3, 3, :))), ':');
        ylabel('Standard deviation of the state error');
        xlabel('Time since launch [s]');
        hold off;
        legend('P1', 'P2', 'P3');
        grid on;
        pause(0.1);
        
        figure(3);
        title('Actual vs. Estimate for Position');
        plot(T, squeeze(x_hat_history(1, 1, :)));
        hold on;
        plot(T, squeeze(x_history(1, 1, :)), '--');
        ylabel('Position');
        xlabel('Time since launch [s]');
        hold off;
        legend('Estimate', 'Actual');
        grid on;
        pause(0.1);
        
        figure(4);
        title('Actual vs. Estimate for Velocity');
        plot(T, squeeze(x_hat_history(2, 1, :)));
        hold on;
        plot(T, squeeze(x_history(2, 1, :)), '--');
        ylabel('Velocity');
        xlabel('Time since launch [s]');
        hold off;
        legend('Estimate', 'Actual');
        grid on;
        pause(0.1);
        
        figure(5);
        title('Actual vs. Estimate for Acceleration');
        plot(T, squeeze(x_hat_history(3, 1, :)));
        hold on;
        plot(T, squeeze(x_history(3, 1, :)), '--');
        ylabel('Acceleration');
        xlabel('Time since launch [s]');
        hold off;
        legend('Estimate', 'Actual');
        grid on;
        pause(0.1);
        
        display_plot = false;
        single_run_plot = false;
    end
    
    disp(['Percent Done: ', num2str(((jdx/number_of_simulations)*100.0))]);
end

error_average = TOTAL_ERROR / number_of_simulations;

res_chk = 0;
for adx = 1:number_of_simulations
    res_chk = res_chk + RESIDUAL_DS(:, 40, adx) * RESIDUAL_DS(:, 500, adx)';
    for bdx = 1:numel(T)
        P_AVG_DS(:, :, bdx) = P_AVG_DS(:, :, bdx) + (ERROR_DS(:, :, bdx, adx) - error_average(:, :, bdx)) * (ERROR_DS(:, :, bdx, adx) - error_average(:, :, bdx))';
    end
end

res_chk = res_chk / number_of_simulations;

P_AVG_DS = P_AVG_DS / (number_of_simulations - 1);
[a,b,c] = size(P_AVG_DS(1, 1, :));

figure(6);
title('Actual Error Variance vs. a priori Error Variance for Position');
for i = 1:c
sqrt_pavg_ds(i) = sqrt(P_AVG_DS(1, 1, i));
sqrt_p_history(i) = sqrt(P_history(1, 1, i));
end
plot(T, sqrt_pavg_ds);
hold on;
plot(T,sqrt_p_history, '--');
ylabel('Position');
xlabel('Time since launch [s]');
legend('Actual Error Variance', 'a priori Error Variance');
grid on;
hold off;
tight_layout();
show();

figure(7);
title('Actual Error Variance vs. a priori Error Variance for Velocity');
plot(T, sqrt(P_AVG_DS(2, 2, :)));
hold on;
plot(T, sqrt(P_history(2, 2, :)), '--');
ylabel('Velocity');
xlabel('Time since launch [s]');
legend('Actual Error Variance', 'a priori Error Variance');
grid on;
hold off;
tight_layout();
show();

figure(8);
title('Actual Error Variance vs. a priori Error Variance for Acceleration');
plot(T, sqrt(P_AVG_DS(3, 3, :)));
hold on;
plot(T, sqrt(P_history(3, 3, :)), '--');
ylabel('Acceleration');
xlabel('Time since launch [s]');
legend('Actual Error Variance', 'a priori Error Variance');
grid on;
hold off;
tight_layout();
show();

