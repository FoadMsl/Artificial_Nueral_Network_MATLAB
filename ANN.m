% =================================================
%       ANN
%       Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
%       Using MATLAB R2022a
% =================================================
clc; clear; close all;

% DataSet ==================================================
    Table = readtable('DataBase.xlsx');
    T = readmatrix('DataBase.xlsx');
    Inputs1 = T(1:180, 1:5); % It should be used as a transposed matrix
    Inputs = Inputs1';
    Target1 = T(1:180 , 7); % It should be used as a transposed matrix
    Target = Target1';
    plot(1:180, Target,'ro', 'LineWidth', 1)    % Plot DataSet Objective
    hold on

% Define Neural Network ======================================
    Num = 41;                      % Number of Test
    NNeron = [3 1 1 3];     % Hidden Layers: NNeron [Layer1  Layer2  ...  LayerN]
    
    % Network Type
%     net = newff(Inputs,Target,NNeron);    % newff: Create a feed-forward backpropagation network. (Form each layer to its next layer)
    net = newcf(Inputs,Target,NNeron);        % newcf: Create a cascade-forward backpropagation network. (From Input to All Layers)
        Outputs = sim(net, Inputs);                     % Neural Network Outputs

% Neural Network Training ====================================
    % === Setups ===
    net.trainFcn = 'trainlm';                          % Levenberg-Marquardt
%     net.trainFcn = 'trainbr';                           % Bayesian Regularization
%     net.trainFcn = 'trainbfg';                         % BFGS Quasi-Newton
%     net.trainFcn = 'trainrp';                           % Resilient Backpropagation
%     net.trainFcn = 'trainscg';                         % Scaled Conjugate Gradient
%     net.trainFcn = 'traincgb';                         % Conjugate Gradient with Powell/Beale Restarts
%     net.trainFcn = 'traincgf';                         % Fletcher-Powell Conjugate Gradient
%     net.trainFcn = 'traincgp';                         % Polak-Ribiére Conjugate Gradient
%     net.trainFcn = 'trainoss';                         % One Step Secant
%     net.trainFcn = 'traingdx';                        % Variable Learning Rate Gradient Descent
%     net.trainFcn = 'traingdm';                       % Gradient Descent with Momentum
%     net.trainFcn = 'traingd';                           % Gradient Descent
    
    net.trainParam.epochs       = 1000;     % Maximum number of epochs to train. The default value is 1000.
    net.trainParam.goal             = 1e-20;    % (Resedual) Performance goal. The default value is 0.
    net.trainParam.max_fail     = 20;          % Maximum validation failures. The default value is 6.
    
    % === Number of NNetwork Training ===
    for i = 1:1
        [net, tr] = train(net,Inputs,Target);           % Train a neural network; [NET,TR] = train(NET,X,T,Xi,Ai,EW); NET: network; TR: training record; X: input data; T: target data; Xi: initial input; Ai: initial layer delays; EW: Error weights.
        Best_Validation_Performance = tr.best_vperf
        Outputs = sim(net,Inputs);                          % Neural Network Training Outputs
        
        plot(1:180, Outputs,'bx', 'LineWidth', 1)  % Plot Results
        hold on

        c = int2str (Num); d = convertCharsToStrings('HW5_ANN'); FigName = d + c;
        save(FigName,'net','tr');                                 % Save Trained Neural Network
        Error = (abs((Target-Outputs)/Target))/180;
        results(i,:) = table(i, Best_Validation_Performance, Error);
    end
    writetable(results, 'ANN_Results.xlsx', 'Sheet', Num);

% =====================================================
% https://www.mathworks.com/help/matlab/import_export/ways-to-import-spreadsheets.html
% https://www.mathworks.com/help/deeplearning/ref/network.html
% https://www.mathworks.com/help/deeplearning/ref/plotperform.html
% https://www.mathworks.com/help/deeplearning/ug/analyze-neural-network-performance-after-training.html
% https://www.mathworks.com/help/deeplearning/ref/feedforwardnet.html#mw_8475ef3a-84a4-4418-bcdc-6502b5d16d54
% https://www.mathworks.com/help/deeplearning/ref/traingdx.html
% https://www.mathworks.com/help/matlab/ref/saveas.html
% =====================================================
% net = newelm(Inputs,Target,NNeron);   % newelm: Create an Elman backpropagation network.
% net.layers{1}.transferFcn='purelin';        % Changing The Activation Function
% view(net)                                                         % View NNetwork Structure
% perf = perform(net,Outputs,Target) % Calculate network performance; perf = perform(net,t,y,ew); net: Input network; t: Network targets; y: Network outputs; ew: Error weights.
% net.trainParam.lr                                   = 0.01;         % Learning rate. The default value is 0.01.
% net.trainParam.lr_inc                            = 1.05;        % Ratio to increase learning rate. The default value is 1.05.
% net.trainParam.lr_dec                           = 0.7;           % Ratio to decrease learning rate. The default value is 0.7.
% net.trainParam.max_perf_inc              = 1.04;        % Maximum performance increase. The default value is 1.04.
% net.trainParam.mc                                 = 0.9;           % Momentum constant. The default value is 0.9.
% net.trainParam.min_grad                     = 1e-16;     % Minimum performance gradient. The default value is 1e-5.
% net.trainParam.show                             = 25;           % Epochs between displays (NaN for no displays). The default value is 25.
% net.trainParam.showCommandLine = false;       % Generate command-line output. The default value is false.
% net.trainParam.showWindow             = true;        % Show training GUI. The default value is true.
% net.trainParam.time                               = inf;           % Maximum time to train in seconds. The default value is inf.