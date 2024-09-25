function [s, f, t] = transform_stft(trace, options)
%transform_stft Generate spectrogram using Short-time Fourier transform.
%   Generate spectrogram using Short-time Fourier transform.

tmp = size(trace);
trace = reshape(trace, tmp(1), []);

% test chensheng
% trace = trace(:, 1);

% perform stft
[s, f, t] = stft(trace, options.sample_rate,...
    'Window', options.window,...
    'OverlapLength', options.overlap_leng,...
    'FFTLength', options.sample_rate / options.freq_res);

% only keep data of desired frequency range.
% the negative and positive part are complex conjugates
s = s(options.sample_rate / options.freq_res / 2:...
    options.sample_rate / options.freq_res / 2 +...
    options.freq_limit / options.freq_res, :, :);
f = f(options.sample_rate / options.freq_res / 2:...
    options.sample_rate / options.freq_res / 2 +...
    options.freq_limit / options.freq_res);

% sub-carrier first
s = permute(s, [3 1 2]);

% abs
s = abs(s);
end

