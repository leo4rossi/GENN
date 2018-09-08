% --- general_NN ---
% Script name: general_NN
% Description: this code implements a general neural network withn c_in
% input neurons and c_out output neurons. The NN has one hidden layer.
% Implemented methodology was taken from the book: "An Introduction to
% Machine Learning", Author: Miroslav Kubat, 2017, Ed: Springer.
% Version: 01.00
% Code author: Leonardo Rossi
% RWTH University - LBB
% Latest update: September 08th, 2018
% Features:
% Commented - Non-optimal - Easy to read - Easy to modify - Modular
% Requirements:
% 1) An input matrix to train the network

clc
close all
clear all

% training data
load data invec outvec
% invec is the input vector, outvec is the ouput vector

[r_in, c_in] = size(invec);
[r_out, c_out] = size(outvec);
in = c_in; % number of input neurons
ou = c_out; % number of output neurons

% Settings and preferences
hi = 7; % number of neurons in the hidden layer
lr = 0.1; % learning rate
% Tuning the learning rate is one of the main problem in training the NN

et = 0.00005; % error trigger

% Max and min for every input dataset
inmax = zeros(in, 1);
inmin = zeros(in, 1);
for y = 1:1:in
    inmax(y, 1) = max( max((invec(:, y))) );
    inmin(y, 1) = min( min((invec(:, y))) );
end % y-for

outmax = max(max(outvec));
outmin = min(min(outvec));

% Data normalisation
iv = zeros(in, r_in);
for y = 1:1:in
    iv(y, :) = ( ( invec(:, y) - inmin(y, 1) )./( inmax(y, 1) - inmin(y, 1) ) )'; % input values (column vectors)
end % y-for
ev = ( ( outvec - outmin )./( outmax - outmin ) )'; % expected values (column vectors)
errv = zeros(ou, 1); % error vector
cont = 1; % continue (boolean)

% Weight marix initialisation
WM1 = -2*rand(in, hi) + 1; % initial values of weights matrix #1
WM2 = -2*rand(hi, ou) + 1; % initial values of weights matrix #2
sums1 = zeros(hi, 1);
hv = zeros(hi, 1); % values computed at the hidden layer
sums2 = zeros(ou, 1);
ov = zeros(ou, 1);
delta1 = zeros(ou, 1);
delta2 = zeros(hi, 1);
ep = 0;
x = 0;

% Training starts
while( cont ) % every cycle is an epoch
    x = x + 1;
    temp = 0; % total error of the epoch
    ep_er = 0; % relative error of the epoch
    ep = ep + 1;
    
    for kk = 1:1:r_in % number of training values
        % forward propagation
        for i = 1:1:hi
            sums1(i, 1) = iv(:, kk)'*WM1(:, i);
            hv(i, 1) = act_fun(sums1(i, 1));
        end % i-for
        
        for j = 1:1:ou
            sums2(j, 1) = hv'*WM2(:, j);
            ov(j, 1) = act_fun(sums2(j, 1));
            errv(j, 1) = ( ev(j, kk) - ov(j, 1) );
        end % i-for
        
        for k = 1:1:ou
            delta1(k, 1) = ov(k, 1)*( 1 - ov(k, 1) )*( ev(j, kk) - ov(k, 1) );
        end % k-for
        
        for g = 1:1:hi
            delta2(g, 1) = hv(g, 1)*( 1 - hv(g, 1) )*( WM2(g, :)*delta1 );
        end % g-for
        
        temp = temp + sum( errv(:, 1) )^2 ;
        
        for i = 1:1:in
            for j = 1:1:hi
                WM1(i, j) = WM1(i, j) + lr*delta2(j, 1)*iv(i, kk);
            end % j-for
        end % i-for
        
        for k = 1:1:hi
            for h = 1:1:ou
                WM2(k, h) = WM2(k, h) + lr*delta1(h, 1)*hv(k, 1);
            end % j-for
        end % i-for
    end % kk-for
    % one epoch ends here
    
    ep_er = temp / r_in
    salva = temp;
    %     drawnow
    hold on
    grid on
    grid minor
    plot(x, ep_er, '--o', 'MarkerFaceColor', 'red')
    if( et > abs(ep_er) )
        cont = 0;
    end % if
    
end % while

% The weight matrixes can be used for prediction.
save ('general_NN','WM1', 'WM2')