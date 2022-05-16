addpath(genpath('FUGIboardMatlab'));
if strcmp(computer, 'MACI64') && strcmp(getenv('DCSCUSB_HEX_FILE'), '')
    setenv('DCSCUSB_HEX_FILE', which('fx2-slave-fifo.hex'));
    disp('MACI64 check done');
end
disp('API loaded');