clear; clc; close all;

set(0, 'DefaultFigureColor', [1 1 1]);
set(0, 'DefaultAxesColor', [1 1 1]);
set(0, 'DefaultTextColor', [0 0 0]);
set(0, 'DefaultAxesXColor', [0 0 0]);
set(0, 'DefaultAxesYColor', [0 0 0]);

datasets = {'Town01','Town02','Town10','KITTI07','MH01','MH03'};
neuro = [148,118,110,75,3,2];
ekf = [235,242,270,85,5,5];
vo = [240,155,100,85,3,20];

imp = [42.6, 51.2, 58.5, 12.6, -2.6, 25.0];
success = [83,17];

figure('Position',[100 100 1000 800]);

subplot(2,2,[1 2]);
bar([neuro;ekf;vo]');
set(gca,'XTickLabel',datasets,'FontName','Times New Roman','FontSize',11);
ylabel('RMSE (m)','FontName','Times New Roman','FontSize',12);
title('(a) RMSE Comparison','FontName','Times New Roman','FontSize',12);
legend('NeuroLocMap','EKF','VO','Location','best','FontName','Times New Roman');
grid on;

subplot(2,2,3);
scatter(imp,1:6,100:10:150,'filled');
yticks(1:6); 
yticklabels(datasets);
xlabel('Improvement vs EKF (%)','FontName','Times New Roman','FontSize',11);
ylabel('Dataset','FontName','Times New Roman','FontSize',11);
title('(b) Improvement','FontName','Times New Roman','FontSize',12);
grid on;

for i = 1:6
    text(imp(i)+1.8, i, sprintf('%.1f%%', imp(i)), ...
         'FontName','Times New Roman','FontSize',10);
end

subplot(2,2,4);
pie(success);
colormap([0.2 0.6 0.9; 0.8 0.4 0.4]);
title('(c) Success Rate: 83%','FontName','Times New Roman','FontSize',12);

print('Fig2_Performance_Comparison','-depsc','-r600');
