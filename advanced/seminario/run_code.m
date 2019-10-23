clc; close all; clearvars;

dataset_path = 'D:/Documents/UFSC/datasets/TRIOS_dataset/';

folders = dir(dataset_path);
folders = folders([folders.isdir]);
folders = {folders(3:end).name};

num_folders = length(folders);

filename = [dataset_path, folders{5}, '/mix.wav'];

[x, fs] = audioread(filename);

fr = 4;
Hop = round(1e-3*fs);
nfft = 4096;
h = hamming(nfft+1);
g1 = 0.6;
g2 = 0.8;

[tfr, ceps, GCoS, upcp, upcpt, upcp_final, t] = CFP_GCoS(x, 4, 44100, 441, hamming(4097), 0.6, 0.8);

imagesc(upcp_final)

