function [MDF] = cal_mdf(window_signal, window_size, sample_rate)
% cal media frequency
% window_signal: signal in sliding window
% window_size: size of sliding window
    % fft
    fft_signal = fft(window_signal);
    % get spectrum amplitude
    P2 = abs(fft_signal/window_size);
    P1 = P2(1:floor(window_size/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    % calculate frequency axis
    f = sample_rate*(0:(window_size/2))/window_size;
    % calculate spectrum enercy
    spectral_energy = cumsum(P1.^2);
    % calculate MDF
    total_energy = spectral_energy(end);
    MDF = f(find(spectral_energy >= total_energy/2, 1));
end