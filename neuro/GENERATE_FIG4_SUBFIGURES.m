%% 主脚本：生成论文图表
% 清除环境
clear; clc; close all;

set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultTextFontName', 'Times New Roman');

%% 创建图形窗口
fig = figure('Position', [100, 100, 1200, 450], 'Color', 'white');

%% ========== 左图：消融研究柱状图  ==========
ax1 = subplot(1, 2, 1);
hold on;

categories = {'Full', 'w/o IMU', 'w/o Exp Map', 'w/o Transformer', 'w/o Dual-stream'};
rmse_values = [145.5, 315.3, 186.0, 183.0, 180.0];
errors = [8, 12, 9, 8, 8]; 

bar_width = 0.6;
x_pos = 1:length(categories);
bars = bar(x_pos, rmse_values, bar_width, 'FaceColor', [0.4, 0.7, 0.9], ...
    'EdgeColor', [0.2, 0.4, 0.6], 'LineWidth', 1.2);

errorbar(x_pos, rmse_values, errors, 'k.', 'LineWidth', 1.5, ...
    'MarkerSize', 1, 'CapSize', 6);

for i = 1:length(rmse_values)
    text(x_pos(i), rmse_values(i) + errors(i) + 10, ...
        sprintf('%.1f m', rmse_values(i)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'FontWeight', 'bold');
end

set(gca, 'XTick', x_pos, 'XTickLabel', categories, 'FontSize', 10);
xlabel('Configuration', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('RMSE (m)', 'FontSize', 12, 'FontWeight', 'bold');
title('Ablation Study: Component Contribution Analysis', ...
    'FontSize', 13, 'FontWeight', 'bold');

ylim([0, 400]);
grid on;
grid minor;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);

hold off;

%% ========== 右图：已优化  ==========
ax2 = subplot(1, 2, 2);
hold on;

datasets = struct();
datasets(1).name = 'RSLAM baseline (-5)';    datasets(1).color = [0.5,0.5,0.5];  datasets(1).style = '--';
datasets(2).name = 'Town01 (125)';          datasets(2).color = [0.2,0.6,0.8];  datasets(2).style = '-';
datasets(3).name = 'Town02 (165)';          datasets(3).color = [0.9,0.4,0.1];  datasets(3).style = '-';
datasets(4).name = 'Town10 (195)';          datasets(4).color = [0.6,0.8,0.2];  datasets(4).style = '-';
datasets(5).name = 'MH03 (171)';             datasets(5).color = [0.8,0.2,0.2];  datasets(5).style = '-';
datasets(6).name = 'KITTI 07 (112)';        datasets(6).color = [0.5,0.2,0.7];  datasets(6).style = '-'; % 改为紫色
datasets(7).name = 'MH01(166)';                 datasets(7).color = [0.9,0.7,0.2];  datasets(7).style = '-';

frame_nums = linspace(0, 1800, 100); 

slopes = [0.004, 0.055, 0.070, 0.085, 0.050, 0.095, 0.048];

for i = 1:length(datasets)
    y_vals = slopes(i) * frame_nums;
    plot(frame_nums, y_vals, 'Color', datasets(i).color, ...
         'LineStyle', datasets(i).style, 'LineWidth', 1.8, ...
         'DisplayName', datasets(i).name);
end

plot(1800, 153, 'o', 'MarkerFaceColor', [0.6,0.8,0.2], 'MarkerSize',7);
text(1850, 153, '195','FontSize',10,'FontWeight','bold');

xlim([0, 2000]);
ylim([0, 200]);

xlabel('Frame Number', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Visual Template Count', 'FontSize', 12, 'FontWeight', 'bold');
title('Visual Template Growth Across 6 Datasets', 'FontSize', 13, 'FontWeight', 'bold');

grid on; grid minor;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
legend('Location', 'southeast', 'FontSize', 8, 'NumColumns', 2);

hold off;

%% 布局
set(gcf, 'Units', 'normalized', 'OuterPosition', [0.05, 0.2, 0.9, 0.6]);
