% =================================================
%       ANN_Prediction
%       Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
%       Using MATLAB R2022a
% =================================================
clc; clear; close all;

%     Num = 41;
for Num = 1:40
    % DataSet ==================================================
    Table = readtable('DataBase.xlsx');
    T = readmatrix('DataBase.xlsx');
    Inputs1 = T(:, 1:5);                            % It should be used as a transposed matrix
    Inputs = Inputs1';
    Target1 = T(: , 7);                               % It should be used as a transposed matrix
    Target = Target1';
    plot(Target,'ro', 'LineWidth', 1)     % Plot DataSet Objective
    hold on

    % Load Trained Nerual Network =================================
    c = int2str (Num); d = convertCharsToStrings('HW5_ANN'); FigName = d + c;
    load(FigName,'net', 'tr');                   % Load Trained Neural Network
    Outputs = sim(net, Inputs);              % Neural Network Outputs
    plot(Outputs,'bx', 'LineWidth', 1)   % Plot Neural Network Outputs
    hold on

    % Prediction ================================================
    Inputs2 = T(180:200, 1:5); % It should be used as a transposed matrix
    P_Inputs = Inputs2'
    Target2 = T(180:200 , 7); % It should be used as a transposed matrix
    P_Target = Target2';
    Prediction = net(P_Inputs);
    
%     Error2 = sum(abs((P_Target-Prediction)/P_Target))/20;
%     results2(Num, :) = table(Num, Error2);
%     writetable(results2, 'ANN_Test_Results.xlsx', 'Sheet', 1);

    plot(180:200, Target2, 'ro', 'LineWidth', 1)              % Plot Results
    plot(180:200, Prediction, 'cx', 'LineWidth', 1)         % Plot Results
    xlabel('Number of Data')
    ylabel('Objective')
    
    
%     pause(1)
%     saveas(gcf, FigName, 'svg')
    hold off
end
    %     plotperf(tr)
    %     e = convertCharsToStrings('PerformancePlot'); PerformancePlot = e + c;
    %     saveas(gcf, PerformancePlot, 'svg')
