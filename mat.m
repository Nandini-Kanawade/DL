
%% Band Power Calculation from EEG signal

clc;
clear all;
close all;


%EEG_data_C3_4 = csvread(‘EEG Data_Eye close.csv');
EEG_data_C3_4 = csvread(‘EEG Data_Meditation.csv');
%EEG_data_C3_4 = csvread(‘EEG Data_Mental Maths.csv');
%EEG_data_C3_4 = csvread(‘EEG Data_Resting state.csv');

V = EEG_data_C3_4(:,1);  

yy1=bandpass(V,[1 60],128);
waveletFunction = 'db4';
[C,L] = wavedec(V,8,waveletFunction); 


D1 = wrcoef('d',C,L,waveletFunction,1); %GAMMA
A1 = wrcoef('a',C,L,waveletFunction,1); %Level 1 Approximation
D2 = wrcoef('d',C,L,waveletFunction,2); %BETA
A2 = wrcoef('a',C,L,waveletFunction,2); %Level 2 Approximation
D3 = wrcoef('d',C,L,waveletFunction,3); %ALPHA
A3 = wrcoef('a',C,L,waveletFunction,3); %Level 3 Approximation
D4 = wrcoef('d',C,L,waveletFunction,4); %THETA 4-7
A4 = wrcoef('a',C,L,waveletFunction,4); %DELTA 1-4 Level 8 Approximation


disp('Power calculation from wavelet coeffs')

pband = bandpower(A4);
disp('delta band power=')
disp(pband)

pband = bandpower(D4);
disp('theta band power=')
disp(pband)

pband = bandpower(D3);
disp('alpha band power=')
disp(pband)

pband = bandpower(D2);
disp('beta band power=')
disp(pband)

pband = bandpower(D1);
disp('gama band power=')
disp(pband)

The most commonly studied waveforms include 
delta (0.5 to 4Hz); <200uV
theta (4 to 7Hz); <200uV meds
alpha (8 to 12Hz); 20-200uV eyesclosed
beta(sigma) (12 to 16Hz) 
beta (13 to 25Hz). <20uV math
gamma (>30Hz) 2uV
