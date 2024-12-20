%% parameters
% sample path
% sample_path = 'D:\论文\我的论文\谢帅论文\data_new\第一批\1\1.3\31.xlsx';
% 500 sampling point per second
sample_rate = 500;
% interference frequency 50 hz of notch filter
interference_freq = 50;
% quality factor of notch filter
quality_factor = 35;
% parameters of bandpasss filter
Fstop1 = 20;          % First Stopband Frequency
Fpass1 = 30;          % First Passband Frequency
Fpass2 = 200;         % Second Passband Frequency
Fstop2 = 250;         % Second Stopband Frequency
Astop1 = 60;          % First Stopband Attenuation (dB)
Apass  = 1;           % Passband Ripple (dB)
Astop2 = 80;          % Second Stopband Attenuation (dB)
match  = 'passband';  % Band to match exactly
% sliding window size
window_size = floor(sample_rate / 2);
% overlap of windows
overlap = floor(window_size / 2);
% hampel flag
is_hampel = true;
% plot flag
plot_figures = false;
% save 
% save_grams = false;
% segementation
is_segement = true;
% segement parameters
segment_len = 1000;
segment_overlap = 500;
segment_start_idx = 500; % 去除前面不平整波形
% stft parameters
stft_options.sample_rate = sample_rate;
stft_options.window_leng = 128;
stft_options.overlap_leng = 64;
stft_options.window = gausswin(stft_options.window_leng);
stft_options.freq_limit = Fpass2;
stft_options.freq_res = 1;
% segment stft parameters
segment_stft_options.sample_rate = sample_rate;
segment_stft_options.window_leng = 50;
segment_stft_options.overlap_leng = 25;
segment_stft_options.window = gausswin(stft_options.window_leng);
segment_stft_options.freq_limit = Fpass2;
segment_stft_options.freq_res = 1;

%% read EMG data
emg_data = readtable(sample_path);
% 一通道采集原始值
channel_one_data = emg_data{:, 3};
% 二通道采集原始值
channel_two_data = emg_data{:, 7};

%% denoise
% hampel filter
if is_hampel
    one_after_hampel = hampel(channel_one_data, 3);
    two_after_hampel = hampel(channel_two_data, 3);
end
% bandpass filter
bp_filter = bandpass_filter(sample_rate, Fstop1, Fpass1, ...
                            Fpass2, Fstop2, Astop1, ...
                            Apass, Astop2, match);
one_after_butter = filter(bp_filter, one_after_hampel);
two_after_butter = filter(bp_filter, two_after_hampel);
% notch filter
notch_filter = designfilt('bandstopiir', 'FilterOrder', 2, ...
                         'HalfPowerFrequency1', interference_freq*(1 - 1/quality_factor), ...
                         'HalfPowerFrequency2', interference_freq*(1 + 1/quality_factor), ...
                         'DesignMethod', 'butter', 'SampleRate', sample_rate);
one_after_notch = filtfilt(notch_filter, one_after_butter);
two_after_notch = filtfilt(notch_filter, two_after_butter);

%% conduct time-frequency analysis and transform to spectrogram
[gram_stft, f_stft, t_stft] = transform_stft(one_after_notch, stft_options);

%% calculate time domain features (MAV, RMS, iEMG) by sliding window method
num_samples = length(channel_one_data);
num_windows = floor((num_samples - window_size) / overlap) + 1;
MAV_one = zeros(num_windows, 1);
RMS_one = zeros(num_windows, 1);
iEMG_one = zeros(num_windows, 1);
MDF_one = zeros(num_windows, 1);
MAV_two = zeros(num_windows, 1);
RMS_two = zeros(num_windows, 1);
iEMG_two = zeros(num_windows, 1);
MDF_two = zeros(num_windows, 1);
for i = 1:num_windows
    % start index and end index of current window
    start_idx = (i - 1) * overlap + 1;
    end_idx = start_idx + window_size - 1;
    % the signals of current window
    window_one = one_after_notch(start_idx:end_idx);
    window_two = two_after_notch(start_idx:end_idx);
    % calculate MAV
    MAV_one(i) = mean(abs(window_one));
    MAV_two(i) = mean(abs(window_two));  
    % calculate RMS
    RMS_one(i) = sqrt(mean(window_one.^2));
    RMS_two(i) = sqrt(mean(window_two.^2));
    % calculate iEMG
    iEMG_one(i) = sum(abs(window_one));
    iEMG_two(i) = sum(abs(window_two));
    % calculate MDF
    MDF_one(i) = cal_mdf(window_one, window_size, sample_rate);
    MDF_two(i) = cal_mdf(window_two, window_size, sample_rate);
end

    overall_RMS_one = sqrt(mean(one_after_notch.^2));
    overall_RMS_two = sqrt(mean(two_after_notch.^2));

%% plot
if plot_figures
    figure(1);
    subplot(4,2,1)
    plot(channel_one_data);
    title("一通道采集原始值");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,2)
    plot(one_after_hampel);
    title("Hampel滤波后一通道数据");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,3)
    plot(one_after_butter);
    title("带通滤波后一通道数据");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,4)
    plot(one_after_notch);
    title("带通滤波及去除工频干扰后一通道数据");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,5)
    plot(MAV_one);
    title("一通道MAV");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,6)
    plot(RMS_one);
    title("一通道RMS");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,7)
    plot(iEMG_one);
    title("一通道iEMG");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,8)
    plot(MDF_one);
    title("一通道MDF");
    xlabel("timestamp");
    ylabel("EMG value");
    figure(2);
    subplot(4,2,1)
    plot(channel_two_data);
    title("二通道采集原始值");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,2)
    plot(two_after_hampel);
    title("Hampel滤波后二通道数据");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,3)
    plot(two_after_butter);
    title("带通滤波后二通道数据");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,4)
    plot(two_after_notch);
    title("带通滤波及去除工频干扰后二通道数据");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,5)
    plot(MAV_two);
    title("二通道MAV");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,6)
    plot(RMS_two);
    title("二通道RMS");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,7)
    plot(iEMG_two);
    title("二通道iEMG");
    xlabel("timestamp");
    ylabel("EMG value");
    subplot(4,2,8)
    plot(MDF_two);
    title("一通道MDF");
    xlabel("timestamp");
    ylabel("EMG value");
    figure(3);
    imagesc(t_stft, f_stft, squeeze(gram_stft(1, :, :)));
    colorbar;
    xlabel("Time (s)");
    ylabel("Frequency (Hz)");
    title("STFT Spectrogram");
end

%% save grams to files
if save_grams
    % cast to float32
    if is_segement
        idx = segment_start_idx:(segment_len - segment_overlap):(length(one_after_notch) - segment_len + 1);
        segments = [];
        for i = 1:length(idx)
            segment_end_idx = idx(i) + segment_len - 1;
            if segment_end_idx <= length(one_after_notch)
                segments(:, i) = one_after_notch(idx(i):segment_end_idx);
                [segment_gram_stft, segment_f_stft, segment_t_stft] = transform_stft(segments(:, i), ...
                    segment_stft_options);
                processed_signals = single(segments(:, i));
                gram_stft = single(segment_gram_stft);
                output_name = strcat(output_dir, sample_name, '_idx', num2str(i), '.mat');
                save(output_name,...
                    'gram_stft', 'processed_signals',...
                    '-v6');
            end
        end
    else
        one_after_notch = single(one_after_notch);
        gram_stft = single(gram_stft);
        output_name = strcat(output_dir, sample_name, '.mat');
    
        save(output_name,...
            'gram_stft', 'one_after_notch',...
            '-v6');
    end
end