clear;
close all;
fclose('all');

%% parameters
save_grams = 1;
plot_figures = false;

root_dir = "D:\论文\我的论文\谢帅论文\data_new\";
main_folder = ["第一批", "第二批", "第三批"];
output_dir = "D:\论文\我的论文\谢帅论文\data_new\processed_data\";
logfile_path = "D:\论文\我的论文\谢帅论文\data_new\emg_data.log";
log_fid = fopen(logfile_path, 'wt');
fprintf(log_fid, '%s\n', datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z'));
person_id = 0;
tic
for i = 1:length(main_folder)
    fprintf(log_fid, "folder %s is processing.\n", main_folder(i));
    fprintf("folder %s is processing.\n", main_folder(i));
    % 跳过'.' 和 '..' 文件夹
    cur_path = fullfile(root_dir, main_folder(i));
    name_folder = dir(cur_path); 
    tmp_path1 = cur_path;
    for j = 1:length(name_folder)
        if strcmp(name_folder(j).name, '.') || strcmp(name_folder(j).name, '..')
            continue;
        end
        person_id = person_id + 1;
        cur_path = fullfile(tmp_path1, name_folder(j).name);
        state_folder = dir(cur_path);
        tmp_path2 = cur_path;
        for k = 1:length(state_folder)
            if strcmp(state_folder(k).name, '.') || strcmp(state_folder(k).name, '..')
                continue;
            end        
            cur_path = fullfile(tmp_path2, state_folder(k).name);
            sample_folder = cur_path;
            sample_folder_char = char(sample_folder);
            state = sample_folder_char(end);
            sample = dir(sample_folder);
            for l = 1:length(sample)
                tmp_path3 = fullfile(sample_folder, sample(l).name);
                [~, ~, ext] = fileparts(tmp_path3);
                if strcmp(ext, '.csv') || strcmp(ext, '.xlsx')
                    sample_path = fullfile(sample_folder, sample(l).name);
                    sample_name = ['volunteer' num2str(person_id) '_' state];
%                     fprintf("%s\n", sample_path);
%                     fprintf("%s\n", sample_name); %% 记得修改，是给数据取名，取名规则personid_状态id
                    get_emg_features;
                end
            end
        end
    end
end
toc
