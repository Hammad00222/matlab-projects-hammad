%% Task 1: Waveform Reconstruction
clc; clear; close all;

% Load Data from Excel File
data = xlsread('Proj_1_data.xls', 'Task1');

% Extract fundamental and harmonic components
f0 = 50; % Fundamental frequency in Hz
T = 1/f0;
ts = linspace(0, 2*T, 256); % Time vector (2 cycles, 128 points per cycle)

% Reconstruct waveform 1
waveform = zeros(size(ts));
for i = 1:size(data,1)
    h = data(i,1);      % Harmonic order
    mag = data(i,2);    % Magnitude
    phase = data(i,3);  % Phase angle (degrees)
    waveform = waveform + mag * cos(2*pi*h*f0*ts + deg2rad(phase));
end

% Reconstruct waveform 2
waveform1 = zeros(size(ts));
for i = 1:size(data,1)
    h = data(i,5);      % Harmonic order
    mag = data(i,6);    % Magnitude
    phase = data(i,7);  % Phase angle (degrees)
    waveform1 = waveform1 + mag * cos(2*pi*h*f0*ts + deg2rad(phase));
end

% Reconstruct waveform 3
waveform2 = zeros(size(ts));
for i = 1:10
    h = data(i,9);      % Harmonic order
    mag = data(i,10);    % Magnitude
    phase = data(i,11);  % Phase angle (degrees)
    waveform2 = waveform2 + mag * cos(2*pi*h*f0*ts + deg2rad(phase));
end

% Plot waveform
figure;
plot(ts, waveform, 'b', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Amplitude');
title('Reconstructed Waveform of 6 pulse variable frequency drive (ASD6)');
% Plot waveform
figure;
plot(ts, waveform1, 'b', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Amplitude');
title('Reconstructed Waveform of 12 pulse DC drive (DCD12)');
% Plot waveform
figure;
plot(ts, waveform2, 'b', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Amplitude');
title('Reconstructed Waveform of 6 pulse thyristor controlled reactor (TCR6)');

%% Task 2: Harmonic Analysis and Power Calculations
clear; clc; close all;

% Parameters
f0 = 50;                % Fundamental frequency (Hz)
fs = 128*f0;            % Sampling frequency (Hz)
samples_per_cycle = 128;
total_cycles = 9;
T = 1/f0;               % Period of fundamental

% Load data
[data, ~, ~] = xlsread('Proj_1_data.xls', 'Tasks2&3');
Vabc = data(:,1:3);     % Voltage phases [Va, Vb, Vc]
Iabc = data(:,4:6);     % Current phases [Ia, Ib, Ic]

%% Part 1: Plot Waveforms
ts = (0:length(Vabc)-1)/fs; % Time vector in seconds

figure;
subplot(2,1,1);
plot(ts, Vabc, 'LineWidth', 1.2);
title('Voltage Waveforms'); xlabel('Time (s)'); ylabel('Voltage (V)');
legend('V_a', 'V_b', 'V_c'); grid on;

subplot(2,1,2);
plot(ts, Iabc, 'LineWidth', 1.2);
title('Current Waveforms'); xlabel('Time (s)'); ylabel('Current (A)');
legend('I_a', 'I_b', 'I_c'); grid on;

%% Part 2: Harmonic Spectrum Analysis
% Extract cycles (2nd = pre, 8th = post)
pre_start = 2*samples_per_cycle + 1;
pre_end = 3*samples_per_cycle;
post_start = 7*samples_per_cycle + 1;
post_end = 8*samples_per_cycle;

% Initialize storage
harmonic_numbers = 1:19;
results = struct();

for phase = 1:3
    % Voltage analysis
    [V_pre_rms, V_pre_spectrum, V_pre_ph] = harmonic_analysis(Vabc(pre_start:pre_end, phase), f0, fs);
    [V_post_rms, V_post_spectrum, V_post_ph] = harmonic_analysis(Vabc(post_start:post_end, phase), f0, fs);

    % Current analysis
    [I_pre_rms, I_pre_spectrum, I_pre_ph] = harmonic_analysis(Iabc(pre_start:pre_end, phase), f0, fs);
    [I_post_rms, I_post_spectrum, I_post_ph] = harmonic_analysis(Iabc(post_start:post_end, phase), f0, fs);

    % Store results
    results(phase).V_pre = V_pre_spectrum;
    results(phase).V_post = V_post_spectrum;
    results(phase).I_pre = I_pre_spectrum;
    results(phase).I_post = I_post_spectrum;

    results(phase).V_pre_phase = V_pre_ph;
    results(phase).I_pre_phase = I_pre_ph;
    results(phase).V_post_phase = V_post_ph;
    results(phase).I_post_phase = I_post_ph;
end

% Display RMS values in tables
display_harmonic_tables(results, harmonic_numbers);

%% Part 3: Harmonic Comparison and Plotting
plot_harmonic_comparison(results, harmonic_numbers);

%% Part 4: Power and THD Calculations

% Compute THD for all three phases
THDV_pre = zeros(1,3);
THDV_post = zeros(1,3);
THDI_pre = zeros(1,3);
THDI_post = zeros(1,3);

% Initialize power variables
P_pre = zeros(1,3);
P_post = zeros(1,3);
Q_pre = zeros(1,3);
Q_post = zeros(1,3);
S_pre = zeros(1,3);
S_post = zeros(1,3);

for phase = 1:3
    % THD Calculation
    THDV_pre(phase) = sqrt(sum(results(phase).V_pre(2:end).^2)) / results(phase).V_pre(1);
    THDV_post(phase) = sqrt(sum(results(phase).V_post(2:end).^2)) / results(phase).V_post(1);
    THDI_pre(phase) = sqrt(sum(results(phase).I_pre(2:end).^2)) / results(phase).I_pre(1);
    THDI_post(phase) = sqrt(sum(results(phase).I_post(2:end).^2)) / results(phase).I_post(1);

    % Power Calculations 
    % Pre-disturbance
    Vh_pre = results(phase).V_pre;    % Harmonic RMS voltages
    Ih_pre = results(phase).I_pre;    % Harmonic RMS currents
    Vph_pre = results(phase).V_pre_phase; % Voltage phase angles
    Iph_pre = results(phase).I_pre_phase; % Current phase angles

    P_pre_phase = 0;
    Q_pre_phase = 0;
    for h = 1:length(Vh_pre)
        theta_diff = Vph_pre(h) - Iph_pre(h); % Phase angle difference
        P_pre_phase = P_pre_phase + Vh_pre(h) * Ih_pre(h) * cos(theta_diff);
        Q_pre_phase = Q_pre_phase + Vh_pre(h) * Ih_pre(h) * sin(theta_diff);
    end
    P_pre(phase) = P_pre_phase;
    Q_pre(phase) = Q_pre_phase;
    S_pre(phase) = rms(Vabc(pre_start:pre_end, phase)) * rms(Iabc(pre_start:pre_end, phase));

    % Post-disturbance
    Vh_post = results(phase).V_post;
    Ih_post = results(phase).I_post;
    Vph_post = results(phase).V_post_phase;
    Iph_post = results(phase).I_post_phase;

    P_post_phase = 0;
    Q_post_phase = 0;
    for h = 1:length(Vh_post)
        theta_diff = Vph_post(h) - Iph_post(h);
        P_post_phase = P_post_phase + Vh_post(h) * Ih_post(h) * cos(theta_diff);
        Q_post_phase = Q_post_phase + Vh_post(h) * Ih_post(h) * sin(theta_diff);
    end
    P_post(phase) = P_post_phase;
    Q_post(phase) = Q_post_phase;
    S_post(phase) = rms(Vabc(post_start:post_end, phase)) * rms(Iabc(post_start:post_end, phase));
end

% Display results
display_power_results(THDV_pre, THDV_post, THDI_pre, THDI_post, P_pre, P_post, Q_pre, Q_post, S_pre, S_post);

%% Task 3: Harmonic Simulation
% Task 3.1: Convert Components to Per-Unit (100 MVA Base) 
clc; clear; close all;

% System Base
S_base = 100; % MVA
f = 50; % Hz
h = [1, 5, 7, 11, 13];

% Line Data (Table 1) Converted to 100 MVA
lines = {
    'UTIL-69', '69-1', 0.00139*10, 0.00296*10;
    'MILL-1', 'GEN1', 0.00122*10, 0.00243*10;
    'MILL-1', 'FDR F', 0.00075*10, 0.00063*10;
    'MILL-1', 'FDR G', 0.00157*10, 0.00131*10;
    'MILL-1', 'FDR H', 0.00109*10, 0.00091*10;
};

% Transformer Data (Table 2) Converted to 100 MVA
transformers = {
    '69-1', 'MILL-1', 15000, 0.4698, 7.9862;
    'GEN1', 'AUX', 1500, 0.9593, 5.6694;
    'FDR F', 'RECT', 1250, 0.7398, 4.4388;
    'FDR F', 'T3 SEC', 1725, 0.7442, 5.9537;
    'FDR G', 'T11 SEC', 1500, 0.8743, 5.6831;
    'FDR H', 'T4 SEC', 1500, 0.8363, 5.4360;
    'FDR H', 'T7 SEC', 3750, 0.4568, 5.4810;
};

for i = 1:size(transformers, 1)
    kVA = transformers{i, 3}; % Transformer rating
    ZR = transformers{i, 4} / 100 * (S_base*1e3 / kVA);
    ZX = transformers{i, 5} / 100 * (S_base*1e3 / kVA);
    transformers{i, 4} = ZR; % Update R
    transformers{i, 5} = ZX; % Update X
end

% Utility, Generator, Capacitor
Z_utility = 0.05 + 1j*1.0; % Already 100 MVA base
Z_gen = 1j * 0.25 * (S_base / 2); % 12.5j pu
Z_c = -1j*16.6666667; 

% Bus Indices (Example Mapping)
bus_names = {'UTIL-69', '69-1', 'MILL-1', 'GEN1', 'AUX', 'FDR F', 'RECT', ...
    'T3 SEC', 'FDR G', 'FDR H', 'T4 SEC', 'T7 SEC', 'T11 SEC'};
nBuses = length(bus_names);
% Display Results

% Display Table Header
fprintf('\n%-10s %-10s %-8s %-8s %-8s %-8s %-8s %-8s\n', ...
    'From', 'To', 'R', 'X_(h=1)', 'X_(h=5)', 'X_(h=7)', 'X_(h=11)', 'X_(h=13)');
fprintf('%s\n', repmat('-', 1, 80));

% Display Data in Table Format for Line Data
for i = 1:size(lines, 1)
    from_bus = lines{i, 1};
    to_bus = lines{i, 2};
    R = lines{i, 3};
    X_base = lines{i, 4};

    % Compute X values for harmonics
    X_values = X_base * h;  

    % Print row
    fprintf('%-10s %-10s %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
        from_bus, to_bus, R, X_values(1), X_values(2), X_values(3), X_values(4), X_values(5));
end

% Display Table Header
fprintf('\n%-10s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n', ...
    'From', 'To', 'kVA', 'R', 'X_(h=1)', 'X_(h=5)', 'X_(h=7)', 'X_(h=11)', 'X_(h=13)');
fprintf('%s\n', repmat('-', 1, 80));

% Display Data in Table Format for transformers data
for i = 1:size(transformers, 1)
    from_bus1 = transformers{i, 1};
    to_bus1 = transformers{i, 2};
    R1 = transformers{i, 4};
    X_base1 = transformers{i, 5};
    rating = transformers{i, 3};
    % Compute X values for harmonics
    X_values1 = X_base1 * h;  

    % Print row
    fprintf('%-10s %-10s %-8.4d %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
        from_bus1, to_bus1, rating, R1, X_values1(1), X_values1(2), X_values1(3), X_values1(4), X_values1(5));
end

% Display Results

disp('Utility Impedance (p.u.):');
disp(Z_utility);

disp('Generator Impedance (p.u.):');
disp(Z_gen);

disp('Capacitor Bank (p.u.):');
disp(Z_c);


%% Task 3.2: Harmonic Analysis

% Harmonic Source Data (Table 3)
harmonics = [
    1,  100.00,  0.00;
    5,   18.24, -55.68;
    7,   11.90, -84.11;
    11,   5.73, -143.56;
    13,   4.01, -175.58;
    17,   1.93,  111.39;
    19,   1.39,  68.30;
    23,   0.94,  -24.61;
    25,   0.86,  -67.64;
    29,   0.71,  -145.46;
    31,   0.62,   176.83;
    35,   0.44,   97.40;
    37,   0.38,   54.36;
];

harmonic_orders = harmonics(:,1);
harmonic_magnitudes = harmonics(:,2) / 100; % Convert to per-unit
harmonic_phases = deg2rad(harmonics(:,3)); % Convert to radians

fs = 5000; % Sampling frequency
t = 0:1/fs:2*(1/50); % Two cycles

% Generate harmonic waveform (superposition)
I_harmonic = zeros(size(t));

for i = 1:length(harmonic_orders)
    I_harmonic = I_harmonic + harmonic_magnitudes(i) * cos(2*pi*harmonic_orders(i)*50*t + harmonic_phases(i));
end

% Plot Harmonic Source Current Waveform
figure;
plot(t, I_harmonic, 'b', 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Current (p.u.)');
title('Harmonic Source Current Waveform');
grid on;

%% Compute Harmonic Power Flow
% Define Fundamental Load Current at Bus 11 (T4 SEC)
S_load = (1150 + 290i) * 1e3; % 1150 kW + j290 kVar
V_bus7 = 0.9756 * exp(1j * deg2rad(-4.68)); % Voltage at Bus 7
I_fund = conj(S_load / (1e6 * S_base)) / V_bus7; % Per-unit

% Generate Harmonic Current Injection
I_inj = containers.Map('KeyType', 'double', 'ValueType', 'any');
for i = 1:size(harmonics, 1)
    h = harmonic_orders(i);
    mag = harmonic_magnitudes(i) * abs(I_fund);
    phase = harmonic_phases(i) + angle(I_fund);
    I_inj(h) = mag * exp(1j * phase);
end

% Solve Harmonic Power Flow
V_harms = containers.Map('KeyType', 'double', 'ValueType', 'any');
for h = harmonic_orders'
    Y = buildYmatrix(h, lines, transformers, Z_utility, Z_gen*h, Z_c/h, bus_names);
    I = zeros(nBuses, 1);
    I(strcmp(bus_names, 'RECT')) = I_inj(h); % Inject at Bus 7
    V = Y \ I;
    V_harms(h) = V;
end

%% Compute THD at Each Bus
THDv = zeros(nBuses, 1);

for bus = 1:nBuses
    V_fund_vector = abs(V_harms(1));  % Get voltage vector for h=1
    V_fund_bus = V_fund_vector(bus);  % Extract voltage at the specific bus
    
    V_dist_sum = 0;  % Initialize sum of harmonic distortions
    for i = 2:length(harmonic_orders)  % Start from 2 (skip fundamental)
        h = harmonic_orders(i);
        if isKey(V_harms, h)
            V_h_vector = abs(V_harms(h));  % Get voltage vector for harmonic h
            V_dist_sum = V_dist_sum + V_h_vector(bus)^2;  % Add squared magnitude
        end
    end
    
    THDv(bus) = sqrt(V_dist_sum) / V_fund_bus * 100;  % Compute THD
end
% Display Voltage THD at Each Bus
fprintf('\nVoltage THD at Each Bus:\n');
for bus = 1:nBuses
    fprintf('%s: %.2f%%\n', bus_names{bus}, THDv(bus));
end

%% Compute Branch Currents at h = 5
Z_lines_new = 0.1 * (100 / S_base); % Per-unit impedance
I_branch_h5 = abs(I_inj(5)) ./ Z_lines_new; % Using per-unit line impedance

% Compute Current THD at each branch
I_THD = sqrt(sum(cellfun(@(h) abs(I_inj(h))^2, num2cell(harmonic_orders(2:end))))) / abs(I_inj(1)) * 100;

% Display Branch Currents at h = 5 and Current THD
fprintf('\nBranch Currents at h=5 (p.u.):\n');
disp(I_branch_h5);
fprintf('Current THD: %.2f%%\n', I_THD);

%% Plot Voltage at Bus 7 for Different Harmonics
figure;
hold on;

bus7_idx = find(strcmp(bus_names, 'RECT')); % Get index for Bus 7
for h = harmonic_orders'
    if isKey(V_harms, h)
        V_h_vector = abs(V_harms(h));  % Get voltage for all buses at harmonic h
        V_bus7_h = V_h_vector(bus7_idx); % Extract voltage for Bus 7
        
        plot(h, V_bus7_h, 'ro', 'MarkerFaceColor', 'r'); % Correctly plot Bus 7 voltage
    end
end

xlabel('Harmonic Order');
ylabel('Voltage Magnitude (p.u.)');
title('Voltage at Bus 7 for Different Harmonics');
grid on;
hold off;

%% Plot Voltage THD Across Buses
figure;
bar(1:nBuses, THDv); % Ensure THDv matches bus count
set(gca, 'XTick', 1:nBuses, 'XTickLabel', bus_names, 'XTickLabelRotation', 45);
xlabel('Bus Name');
ylabel('THD (%)');
title('Voltage THD at Each Bus');
grid on;
%% Task 3.3: Network Frequency Response
h_steps = 1:0.5:19; % Frequency steps (Δh=0.5)
Z_bus7 = zeros(size(h_steps));

for idx = 1:length(h_steps)
    h = h_steps(idx);
    Y = buildYmatrix(h, lines, transformers, Z_utility, Z_gen*h, Z_c/h, bus_names);
    I_test = zeros(nBuses, 1);
    I_test(strcmp(bus_names, 'RECT')) = 1; % 1 p.u. injection
    V_test = Y \ I_test;
    Z_bus7(idx) = V_test(strcmp(bus_names, 'RECT')); % Impedance = V/I
end

% Plot Frequency Response
figure;
plot(h_steps, abs(Z_bus7));
xlabel('Harmonic Order');
ylabel('Impedance Magnitude (p.u.)');
title('Frequency Response at Bus 7 (RECT)');
%% Helper Function: Build Y Matrix
function Y = buildYmatrix(h, lines, transformers, Z_utility, Z_gen, Z_c, bus_names)
    nBuses = length(bus_names);
    Y = zeros(nBuses, nBuses, 'like', 1j);
    
    % Add Lines
    for i = 1:size(lines, 1)
        from = find(strcmp(bus_names, lines{i,1}));
        to = find(strcmp(bus_names, lines{i,2}));
        R = lines{i,3};
        X = lines{i,4} * h; % Frequency scaling
        Z = R + 1j*X;
        Y(from, to) = Y(from, to) - 1/Z;
        Y(to, from) = Y(to, from) - 1/Z;
        Y(from, from) = Y(from, from) + 1/Z;
        Y(to, to) = Y(to, to) + 1/Z;
    end

    % Add Transformers
    for i = 1:size(transformers, 1)
        from = find(strcmp(bus_names, transformers{i,1}));
        to = find(strcmp(bus_names, transformers{i,2}));
        ZR = transformers{i,4};
        ZX = transformers{i,5} * h; % Frequency scaling
        Z_tr = ZR + 1j*ZX;
        Y(from, to) = Y(from, to) - 1/Z_tr;
        Y(to, from) = Y(to, from) - 1/Z_tr;
        Y(from, from) = Y(from, from) + 1/Z_tr;
        Y(to, to) = Y(to, to) + 1/Z_tr;
    end

    % Utility (Bus 1)
    util_bus = find(strcmp(bus_names, 'UTIL-69'));
    Y(util_bus, util_bus) = Y(util_bus, util_bus) + 1/Z_utility;

    % Generator (GEN1)
    gen_bus = find(strcmp(bus_names, 'GEN1'));
    Y(gen_bus, gen_bus) = Y(gen_bus, gen_bus) + 1/Z_gen;

    % Capacitor (FDR F)
    cap_bus = find(strcmp(bus_names, 'FDR F'));
    Y(cap_bus, cap_bus) = Y(cap_bus, cap_bus) + 1/Z_c;
end

%% Helper functions for task 2
function [rms_total, spectrum, phase_angles] = harmonic_analysis(signal, f0, fs)
    N = length(signal);
    Y = fft(signal);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    angles = angle(Y(1:N/2+1)); % Get phase angles

    % Harmonic extraction
    spectrum = zeros(1,19);
    phase_angles = zeros(1,19);
    for h = 1:19
        bin = h + 1; % FFT bin for h-th harmonic
        spectrum(h) = P1(bin)/sqrt(2); % RMS value
        phase_angles(h) = angles(bin);  % Phase angle
    end
    rms_total = rms(signal); % Total RMS including all harmonics
end

function plot_harmonic_comparison(results, harmonics)
    phases = {'A', 'B', 'C'};
    for phase = 1:3
        figure;
        subplot(2,1,1);
        bar(harmonics, [results(phase).V_pre; results(phase).V_post]');
        title(['Phase ', phases{phase}, ' Voltage Harmonics']);
        xlabel('Harmonic Order'); ylabel('Magnitude (% of Fundamental)');
        legend('Pre-Disturbance', 'Post-Disturbance');

        subplot(2,1,2);
        bar(harmonics, [results(phase).I_pre; results(phase).I_post]');
        title(['Phase ', phases{phase}, ' Current Harmonics']);
        xlabel('Harmonic Order'); ylabel('Magnitude (% of Fundamental)');
        legend('Pre-Disturbance', 'Post-Disturbance');
    end
end


function display_power_results(THDV_pre, THDV_post, THDI_pre, THDI_post, P_pre, P_post, Q_pre, Q_post, S_pre, S_post)
    fprintf('\n=== Power and THD Results ===\n');
    fprintf('%-10s %-15s %-15s\n', 'Phase', 'Pre-Disturbance', 'Post-Disturbance');
    for phase = 1:3
        fprintf('\nPhase %c:\n', 'A'+phase-1);
        fprintf('THD_V:    %-6.2f%%      %-6.2f%%\n', THDV_pre(phase)*100, THDV_post(phase)*100);
        fprintf('THD_I:    %-6.2f%%      %-6.2f%%\n', THDI_pre(phase)*100, THDI_post(phase)*100);
        fprintf('P:        %-6.2f W      %-6.2f W\n', P_pre(phase), P_post(phase));
        fprintf('Q:        %-6.2f VAR    %-6.2f VAR\n', Q_pre(phase), Q_post(phase));
        fprintf('S:        %-6.2f VA     %-6.2f VA\n', S_pre(phase), S_post(phase));
    end
end

function display_harmonic_tables(results, harmonics)
    fprintf('\n=== Harmonic Spectrum Results ===\n');
    phases = {'A', 'B', 'C'};
    for phase = 1:3
        fprintf('\nPhase %s:\n', phases{phase});
        fprintf('Harmonic Order   Pre      Post \n');
        fprintf('%6d              %8.2f    %8.2f\n', [harmonics; results(phase).V_pre; results(phase).V_post]);
    end
end

