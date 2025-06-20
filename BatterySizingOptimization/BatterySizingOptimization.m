clc;
clear all;
close all;

% Load profile
data = readtable('Load_for_2025.csv');
load_profile = data.Load_Forecasted;
load_profile = load_profile(2162:2881);
% load_profile = load_profile(5762:6481);
% Co-gen
data2 = readtable('co_generation.csv');
time = datetime(data2.DateTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
cogen_profile = data2.CoGenMW(8762:end-23,:) * 1000;
cogen_profile = cogen_profile(2162:2881);
% PV
data3 = readtable('solar_generation_thailand.csv');

% Convert timestamp to datetime
data3.timestamp = datetime(data3.timestamp, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

% Extract hour from the timestamp
data3.hour = hour(data3.timestamp);

solar_profile = data3.solar_generation_MW(8762:end-24,:) * 1000;
solar_profile = solar_profile(2162:2881);

% TOU
data4 = readtable('Extended_TOU_Rates.csv');
tou_rates = data4.RatebahtkWh(1:end-24,:);
tou_rates = tou_rates(2162:2881);


%%
% Constants (all in kW)
peak_demand_charge = 74.14; % Baht/kWh for one day
Peak_d_charges=(max(load_profile)*peak_demand_charge);
max_g_cost = sum(3000 .* tou_rates);
max_grid_demand=3000;
max_solar = 5600; % kW
max_cogen = 7800; % kW
time_steps = length(load_profile); % 8760-hour period

% Define battery sizes to evaluate
battery_sizes = linspace(0, 10000, 100); % kWh
costs = zeros(size(battery_sizes));
costs1 = zeros(size(time_steps));
% c_rate = [0.05 0.1 0.2 0.5 1 2 5]; % 1C rate
c_rate = 1; % 1C rate
% for j=1:length(c_rate)
% Problem Definition
    nVar = 1;  % Number of Decision Variables (Battery Size)
    VarSize = [1, nVar];  % Size of Decision Variables Matrix
    VarMin = 0;  % Minimum Battery Size
    VarMax = 10000;  % Maximum Battery Size (in kWh)

    % PSO Parameters
    MaxIt = 100;  % Maximum Number of Iterations
    nPop = 50;    % Population Size (Swarm Size)
    w = 1;      % Inertia Weight
    wdamp = 0.99; % Inertia Weight Damping Ratio
    c1 = 2;       % Personal Learning Coefficient
    c2 = 2;       % Global Learning Coefficient
    BestCosts1 = zeros(MaxIt, 1);
    

% for j=1:length(c_rate)
    % Initialization
    particle = struct();
    globalBest.cost = inf;

    for i = 1:nPop
        particle(i).position = unifrnd(VarMin, VarMax, VarSize);
        particle(i).velocity = zeros(VarSize);
        [particle(i).cost,grid_cost,Initial_cost,annual_maintenance,annual_replace,PCS_cost] = objectiveFunction(particle(i).position, solar_profile, cogen_profile, load_profile, tou_rates, peak_demand_charge, c_rate);
        % particle(i).cost = cost_function(particle(i).position);

        % Update Personal Best
        particle(i).best.position = particle(i).position;
        particle(i).best.cost = particle(i).cost;

        % Update Global Best
        if particle(i).best.cost < globalBest.cost
            globalBest = particle(i).best;
        end
    end

% Array to hold best cost value on each iteration
    BestCosts = zeros(MaxIt, 1);
    BestSizes = zeros(MaxIt, 1);
% while(1)
    % PSO Main Loop
    for it = 1:MaxIt
        for i = 1:nPop
            % Update Velocity
            particle(i).velocity = w*particle(i).velocity ...
                                   + c1*rand(VarSize).*(particle(i).best.position - particle(i).position) ...
                                   + c2*rand(VarSize).*(globalBest.position - particle(i).position);

            % Update Position
            particle(i).position = particle(i).position + particle(i).velocity;

            % Apply Bounds
            particle(i).position = max(particle(i).position, VarMin);
            particle(i).position = min(particle(i).position, VarMax);


            % Evaluation
            [particle(i).cost,grid_cost,Initial_cost,annual_maintenance,annual_replace,PCS_cost] = objectiveFunction(particle(i).position, solar_profile, cogen_profile, load_profile, tou_rates, peak_demand_charge, c_rate);
            % particle(i).cost = cost_function(particle(i).position);
            % Update Personal Best
            if particle(i).cost < particle(i).best.cost
                particle(i).best.position = particle(i).position;
                particle(i).best.cost = particle(i).cost;

                % Update Global Best
                if particle(i).best.cost < globalBest.cost
                    globalBest = particle(i).best;
                end
            end
        end

        % Update Inertia Weight
        w = w * wdamp;

        % Display Iteration Information
        disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(globalBest.cost)]);
        
        % Store the Best Cost Value
        BestCosts(it) = min(globalBest.cost);
        BestSizes(it) = min(globalBest.position);
        
    end

    % Output Result
    disp(['Optimal Battery Size: ' num2str(globalBest.position) ' kWh']);
    disp(['Minimum Cost: ' num2str(globalBest.cost)]);
    [costsb,grid_cost,Initial_cost,annual_maintenance,annual_replace,PCS_cost] = objectiveFunction(0, solar_profile, cogen_profile, load_profile, tou_rates, peak_demand_charge, c_rate);
    
    disp(['Grid Import Cost (Monthly): ' num2str(grid_cost)]);
    disp(['Max allowable Grid Import Cost (Monthly): ' num2str(max_g_cost)]);
    disp(['Capital Cost of Battery: ' num2str(Initial_cost)]);
    disp(['Capital Cost of PCS: ' num2str(PCS_cost)]);
    disp(['Annual replacement cost of battery (scaled down to Monthly): ' num2str(annual_replace)]);
    disp(['Annual maintenance cost of battery (scaled down to Monthly): ' num2str(annual_maintenance)]);
    disp(['Peak demand charges (Monthly): ' num2str(Peak_d_charges)]);
    disp(['Total: ' num2str(grid_cost+Initial_cost+annual_replace+Peak_d_charges+annual_maintenance+PCS_cost)]);
    disp(['adjusted grid cost: ' num2str(globalBest.cost-(Initial_cost+annual_replace+Peak_d_charges+annual_maintenance+PCS_cost))]);
    disp(['Battery Power for 1 Hour: ' num2str(globalBest.position*c_rate) ' kW']);


    % Plot the Best Cost per Iteration
    figure;
    plot(BestCosts, 'LineWidth', 2);
    xlabel('Iteration','FontSize', 16);
    ylabel('Best Cost (Baht)','FontSize', 16);
    title('PSO Progress: Best Cost by Iteration for 1C','FontSize', 16);
    % legend('0.05C','0.1C','0.2C','0.5C','1C','2C','5C')
    set(gca, 'FontSize', 14);
    grid on;
    % hold on;

    % Plot the Best Cost per Iteration
    figure;
    plot(BestSizes,'color','r', 'LineWidth', 2);
    xlabel('Iteration','FontSize', 16);
    ylabel('Best Sizes (kWh)','FontSize', 16);
    title('PSO Progress: Best Sizes of Batteries by Iteration for 1C','FontSize', 16);
    set(gca, 'FontSize', 14);
    % legend('0.05C','0.1C','0.2C','0.5C','1C','2C','5C')
    grid on;
    % hold on;
   
% end
c_rate=1;
% Evaluate the objective function for each battery size
for i = 1:length(battery_sizes)
    [costs1,grid_cost,Initial_cost,annual_maintenance,annual_replace,PCS_cost] = objectiveFunction(battery_sizes(i), solar_profile, cogen_profile, load_profile, tou_rates, peak_demand_charge, c_rate);
    costs(i)=sum(costs1);
end

% Plot the results
figure;
plot(battery_sizes, costs, 'b-', 'LineWidth', 2);
xlabel('Battery Size (kWh)','FontSize', 16);
ylabel('Total System Cost (Baht)','FontSize', 16);
title('Objective Function: Total System Cost vs. Battery Size','FontSize', 16);
set(gca, 'FontSize', 14);
grid on;



% Example usage

c_rate = 1; % 1C rate
avg_discharge_hours = calculateDischargeHours(globalBest.position, solar_profile, cogen_profile, load_profile, c_rate);
disp(['Average Discharge Hours per Day: ', num2str(avg_discharge_hours)]);

function avg_discharge_hours = calculateDischargeHours(battery_size, solar_profile, cogen_profile, load_profile, c_rate)
    % Initialize parameters
    BESS_max = battery_size; % Max capacity in kWh
    BESS_min = 0.2 * BESS_max; % Min capacity to maintain in kWh (e.g., 20%)
    SOC = 0.5 * BESS_max; % Starting at 50% of capacity
    max_discharge_power = BESS_max * c_rate; % Max discharge power based on C-rate
    max_charge_power = BESS_max * c_rate;
    discharge_hours = 0;
    total_hours = 0;

    % Simulate for the given number of days

        for t = 1:length(load_profile)
            % Current generation and load
            E_pv = solar_profile(t);
            E_cogen = cogen_profile(t);
            E_load = load_profile(t);

            % Net energy
            Enet = E_pv + E_cogen - E_load;

            if Enet < 0
                % Energy deficit, battery discharge needed
                dischargeEnergy = min(-Enet, min(SOC - BESS_min, max_discharge_power));
                SOC = SOC - dischargeEnergy; % Discharge battery
                if dischargeEnergy > 0
                    discharge_hours = discharge_hours + 1;
                end
            else
                % Energy surplus, battery charge
                chargeEnergy = min(Enet, min(BESS_max - SOC, max_charge_power));
                SOC = SOC + chargeEnergy; % Charge battery
            end

            % Track total hours
            total_hours = total_hours + 1;
        end
        avg_discharge_hours = discharge_hours;
end
    
    

function [cost,Grid_cost,initialCost,annualMaintenanceCost,totalReplacementCost,PCS_cost] = objectiveFunction(battery_size, solar_profile, cogen_profile, load_profile, tou_rates, peak_demand_charge, c_rate)
    BESS_cost_per_kWh = 5800; % Updated cost per kWh of battery capacity in Thai Baht
    OandM_factor = 0.07; % Increased annual maintenance cost as a percentage of initial capital
    replacement_factor = 0.02; % Increased cost factor for battery replacement over its lifetime
    
    % Scaling factors for costs
    % scaling_factor = 0.001; % Scale down the costs by 0.1% to balance with grid costs

    % Scaled costs
    % scaled_BESS_cost_per_kWh = BESS_cost_per_kWh * scaling_factor;
    % scaled_OandM_factor = OandM_factor * scaling_factor;
    % scaled_replacement_factor = replacement_factor * scaling_factor;


            max_grid_cost = sum(3000 .* tou_rates);
            BESSSize = battery_size;  % Example size in kWh
            BESS_max = BESSSize;  % Max capacity in kWh
            BESS_min = 0.2 * BESSSize;  % Min capacity to maintain in kWh, for example, 20%
            SOC = 0.5 * BESSSize;  % Starting at 50% of capacity

            Grid_cost = 0;
            penalty = 0;
    for t = 1:length(load_profile)
        % Current generation and load
                E_pv = solar_profile(t);
                E_cogen = cogen_profile(t);
                E_load = load_profile(t);
                tou = tou_rates(t);
                % Maximum charge and discharge power based on C-rate
        max_charge_power = BESSSize * c_rate; % kW
        max_discharge_power = BESSSize * c_rate; % kW
                % Net energy
                Enet = E_pv + E_cogen - E_load;

                % Islanded mode logic
                if Enet > 0
                    % Excess energy
                    if SOC < BESS_max
                        % SOC = min(SOC + Enet * charging_efficiency, BESS_max);
                        % chargeEnergy = min(Enet, BESS_max - SOC)* c_rate;
                        chargeEnergy = min(Enet, min(BESS_max - SOC, max_charge_power));
                        SOC = SOC + chargeEnergy;

                    end
                elseif Enet < 0
                    if E_pv + E_cogen + SOC >= E_load
                        if SOC > BESS_min
                        % Discharge battery, but don't go below the min capacity
                        % dischargeEnergy = min(-Enet, SOC - BESS_min)* c_rate;
                        dischargeEnergy = min(-Enet, min(SOC - BESS_min, max_discharge_power));
                        SOC = SOC - dischargeEnergy;

                        end 
                    else
                        if E_pv>0
                        % chargeEnergy = min(E_pv, BESS_max - SOC)* c_rate;
                        chargeEnergy = min(E_pv, min(BESS_max - SOC, max_charge_power));
                        SOC = SOC + chargeEnergy;
                        end
                        energy_needed_from_grid = abs(E_cogen - E_load);
                        energy_cost=energy_needed_from_grid*tou;
                        if Grid_cost + energy_cost <= max_grid_cost
                            if energy_needed_from_grid<=3000
                                Grid_cost = Grid_cost + energy_cost;
                            else
                                penalty = penalty + 800;
                                Grid_cost = Grid_cost + energy_cost + penalty;
                            end
                        else
                         penalty = penalty + 800;
                         Grid_cost = Grid_cost + energy_cost + penalty;
                          
                        end                        
                    end

                end
    end

    % Adjusted costs taking efficiency and degradation into account
    % 
    initialCost = BESSSize * BESS_cost_per_kWh;
    annualMaintenanceCost = (BESSSize * BESS_cost_per_kWh * OandM_factor)/12;
    totalReplacementCost = (BESSSize * BESS_cost_per_kWh * replacement_factor)/12;
    % 
    % Calculate total costs
    % initialCost = BESSSize * scaled_BESS_cost_per_kWh;
    % annualMaintenanceCost = BESSSize * scaled_BESS_cost_per_kWh * scaled_OandM_factor;
    % totalReplacementCost = BESSSize * scaled_BESS_cost_per_kWh * scaled_replacement_factor;

    % Summing all costs including disposal and subtracting savings

    % cost = Grid_cost + initialCost + annualMaintenanceCost + totalReplacementCost + ((max(load_profile)*peak_demand_charge)/30) ;
    if BESSSize >= 100 && BESSSize <500
        PCS_cost = BESSSize*300*36.26;
    elseif BESSSize >= 500 && BESSSize <1000
        PCS_cost = BESSSize*150*36.26;
    elseif BESSSize >= 1000 && BESSSize <2000
        PCS_cost = BESSSize*150*36.26;
    elseif BESSSize >= 2000 && BESSSize <5000
        PCS_cost = BESSSize*150*36.26;
    elseif BESSSize >= 5000 && BESSSize <10000
        PCS_cost = BESSSize*120*36.26;
    else
        PCS_cost = BESSSize*75*36.26;
    end
    % Summing all costs
    scaled_cost = Grid_cost + initialCost + annualMaintenanceCost + totalReplacementCost + PCS_cost;
    % scaled_cost = scaled_cost + (max(load_profile) * peak_demand_charge) / 30; % Include peak demand charge

    % Bring the scaled cost back to original values
    % cost = scaled_cost / scaling_factor;
    % cost = cost + ((max(load_profile)*peak_demand_charge)/30);
    cost = scaled_cost+(max(load_profile)*peak_demand_charge);

end