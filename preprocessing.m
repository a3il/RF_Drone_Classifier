close all; clear; clc

load_filename = 'D:\IIT PATNA\DroneRF';
save_filename = 'D:\IIT PATNA\result';

BUI{1,1} = {'00000'};
BUI{1,2} = {'10000','10001','10010','10011'};
BUI{1,3} = {'10100','10101','10110','10111'};
BUI{1,4} = {'11000'};
M = 2048;
L = 1e5;
Q = 10;

for opt = 1:length(BUI)
    for b = 1:length(BUI{1,opt})
        if(strcmp(BUI{1,opt}{b},'00000'))
            N = 40;
        elseif(strcmp(BUI{1,opt}{b},'10111'))
            N = 17;
        else
            N = 20;
        end
        data = [];
        cnt = 1;
        for n = 0:N
            x = csvread([load_filename '\' BUI{1,opt}{b} 'L_' num2str(n) '.csv']);
            y = csvread([load_filename '\' BUI{1,opt}{b} 'H_' num2str(n) '.csv']);
            for i = 1:length(x)/L
                st = 1 + (i-1)*L;
                fi = i*L;
                xf = abs(fftshift(fft(x(st:fi)-mean(x(st:fi)),M))); xf = xf(end/2+1:end);
                yf = abs(fftshift(fft(y(st:fi)-mean(y(st:fi)),M))); yf = yf(end/2+1:end);
                data(:,cnt) = [xf ; (yf*mean(xf((end-Q+1):end))./mean(yf(1:Q)))];
                cnt = cnt + 1;
            end
        end
        Data = data.^2;
        save([save_filename '\' BUI{1,opt}{b} '.mat'],'Data');
    end
end
