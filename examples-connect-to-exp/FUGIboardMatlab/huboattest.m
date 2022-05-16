% Find the hex file to download to the setup (only required for Mac)
if strcmp(computer, 'MACI64') && strcmp(getenv('DCSCUSB_HEX_FILE'), '')
    setenv('DCSCUSB_HEX_FILE', which('fx2-slave-fifo.hex'));
end;

%clear xpos1 xpos2 xcurr xvolt1 xvolt2;
fugiboard('CloseAll');
h=fugiboard('Open', 'huboat1');
h.WatchdogTimeout = 0.5;
fugiboard('SetParams', h);
fugiboard('Write', h, 0, 0, [0 0], [0 0]);    % dummy write to sync interface board
fugiboard('Write', h, 4+1, 1, [0 0], [0 0]);  % reset position, activate relay

data = fugiboard('Read', h); % get version info from FPGA
model = bitshift(data(1), -4);
version = bitand(data(1), 15);
disp(sprintf('FPGA setup %d, version %d', model, version));

fugiboard('Write', h, 0, 1, [0 0], [0 0]);  % end reset
pause(0.1); % give relay some time to act
Ts = 0.012; %'sample time'
steps = 1000;
xstat = zeros(1,steps);
xpos1 = zeros(1,steps);
xpos2 = zeros(1,steps);
xcspd1 = zeros(1,steps);
xcspd2 = zeros(1,steps);
xcurr = zeros(1,steps);
xvolt1 = zeros(1,steps);
xvolt2 = zeros(1,steps);
xreltime = zeros(1,steps);
xabstime = zeros(1,steps);
tic;
bt = toc;
for X=1:steps
    Y=(X/2);
    fugiboard('Write', h, 0, 1, [10*sin(Y/50) 1.0], [+Y -Y +Y +(Y/2) +500*sin(Y/10) +500*sin(Y/5) +500*sin(Y/2) +500*sin(Y)]);
    %pause(0.020);
    %fugiboard('Write', h, 0, 1, [10*sin(Y/50) 1.0], [5 +500*sin(Y/10); 6 +500*sin(Y/5); 7 +500*sin(Y/2); 8 +500*sin(Y)]);
    data = fugiboard('Read', h);
    xstat(X) = data(1);
    xreltime(X) = data(2);
    xpos1(X) = data(3);
    xpos2(X) = data(4);
    xcurr(X) = data(5);
    xvolt1(X) = data(6);
    xvolt2(X) = data(7);
    xcspd1(X) = data(8);
    xcspd2(X) = data(9);
    t = bt + (Ts * X);
    while (toc < t); end;
end
toc;
% shutdown outputs
fugiboard('Write', h, 0, 0, [0 0], [63 0]);

figure(1); stairs([xpos1; xpos2]'); ylabel('Position (P1,P2)'); axis([0 steps 0 1000]);
figure(2); stairs([xcspd1; xcspd2]'); ylabel('Calculated Speed (V1, V2)'); axis([0 steps 0 1000]);
% figure(3); stairs([xvolt1; xvolt2]'); ylabel('X, Y'); axis([0 steps -45.0 45.0 ]);
figure(4); stairs(xcurr); ylabel('Current(A01)'); axis([0 steps -5.1 5.1 ]);
