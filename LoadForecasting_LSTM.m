clc;
clear all;
close all;
% Load Historical Load Data

data = readtable('factory_hourly_data_smoothed.csv');
% data.DateTime = datetime(data.DateTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

% Extract and Normalize Load Data
% time = datetime(data.Date);
time = datetime(data.DateTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
% load = data.Adjusted_Load;
load = data.Load2;
 
loadNormalized = (load - mean(load)) / std(load);

% Prepare Data for LSTM
X = loadNormalized(1:end-1)';
Y = loadNormalized(2:end)';
XTrain = reshape(X, [1, length(X), 1]);  % 1 feature, sequence length, 1 sequence
YTrain = reshape(Y, [1, length(Y), 1]);  % Same for Y


% Define LSTM Network Architecture
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 256; % Adjusted number of hidden units

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 75, ...  % Adjusted number of epochs %until now 50 epochs give best results
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 100, ...  % Adjusted drop period
    'LearnRateDropFactor', 0.1, ...  % Adjusted drop factor
    'Verbose', 1, ...
    'Plots', 'training-progress');

% Train the LSTM Network
net = trainNetwork(XTrain, YTrain, layers, options);

% Forecast Future Load
% numTimeStepsTest = 365*2*24; % Number of hours in 2 years for hourly data
numTimeStepsTest = size(load); % Number of hours in 2 years for hourly data
XTest = loadNormalized(end-numTimeStepsTest+1:end-1);
XTest = reshape(XTest,[1, length(XTest), 1]);

YPred = predict(net, XTest, 'MiniBatchSize', 1);

% Unnormalize the Predictions
YPred = YPred*std(load) + mean(load);

layers1 = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Training Options
options1 = trainingOptions('adam', ...
    'MaxEpochs', 75, ...  % Adjusted number of epochs
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 100, ...  % Adjusted drop period
    'LearnRateDropFactor', 0.1, ...  % Adjusted drop factor
    'Verbose', 1, ...
    'Plots', 'training-progress');

% Train the LSTM Network
net1 = trainNetwork(XTrain, YTrain, layers1, options1);

% Forecast Future Load
% numTimeStepsTest = 365*2*24; % Number of hours in 2 years for hourly data
numTimeStepsTest = size(load); % Number of hours in 2 years for hourly data
XTest = loadNormalized(end-numTimeStepsTest+1:end-1);
XTest = reshape(XTest,[1, length(XTest), 1]);

YPred1 = predict(net1, XTest, 'MiniBatchSize', 1);

% Unnormalize the Predictions
YPred1 = YPred1*std(load) + mean(load);

% Assuming 'time' is a datetime array from your CSV data
lastDate = time(end); % The last date (and time) in your dataset

% Calculate the number of hours for the forecast period
% numHoursForecast = 17520; % Total hours in 2 years
numHoursForecast = size(load); % Total hours in 2 years

% Create a datetime array for the forecast period, starting an hour after 'lastDate'
timeTest = lastDate + hours(1):hours(1):lastDate + hours(numHoursForecast(1,1));

figure
plot( YPred1,'Color',[0.9290 0.6940 0.1250]) % Use the entire timeTest array for plotting
xlabel('Time (Hours)')
ylabel('Load (MW)')
title('2-Year Hourly Load Forecast using LSTM')
hold on
plot( YPred)
plot( load(2:end)','Color','b')
legend('LSTM','Bi-LSTM','Actual')
% Assuming you have two arrays: 'actualValues' and 'predictedValues'
% These should be column vectors of equal length
actualValues = load(2:end)';% Your actual values here;
predictedValues = YPred;% Your LSTM model predictions here;

% Plot actual vs. predicted values
figure;
scatter(actualValues, predictedValues, 'filled');
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs. Predicted Values');
hold on;

% Fit a linear regression line to the actual vs. predicted values
coeffs = polyfit(actualValues, predictedValues, 1);
% Create a line from the linear regression coefficients
xFit = linspace(min(actualValues), max(actualValues), 100);
yFit = polyval(coeffs, xFit);

% Plot the linear regression line
plot(xFit, yFit, 'r-', 'LineWidth', 2);
legend('Actual vs. Predicted', 'Linear Fit', 'Location', 'best');

% Calculate and display the R-squared value to assess the fit
yResid = predictedValues - polyval(coeffs, actualValues);
SSResid = sum(yResid.^2);
SSTotal = (length(predictedValues)-1) * var(predictedValues);
rsq = 1 - SSResid/SSTotal;
disp(['R-squared value: ', num2str(rsq)]);

% Add y = x line to compare
plot(xFit, xFit, 'k--');
legend('Actual vs. Predicted', 'Linear Fit', 'y = x', 'Location', 'best');

hold off;






