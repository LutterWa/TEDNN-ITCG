clear
close all
load('En_test.mat')
% y_hat(9833,:)=y_hat(9832,:);
% y_en(9833,:)=y_en(9832,:);
error = y-y_hat;
error_en = y-y_en;
MSE = mean(error.^2);
MAE = mean(abs(error));
RMSE = sqrt(MSE);
mse_en = mean(error_en.^2);
mae_en = mean(abs(error_en));
rmse_en = sqrt(mse_en);
r2 = 1-(sum(error.^2)/sum((y-mean(y)).^2));
r2_en = 1-(sum(error_en.^2)/sum((y-mean(y)).^2));

figure,hold on;
plot(rmse_en,'LineWidth',2)
plot(mae_en,'LineWidth',2)
grid on;
legend('RMSE','MAE','Interpreter','latex')
xlabel('Epochs','Interpreter','latex')

figure,hold on;grid on;
plot(error(:,5),'--')
plot(error_en(:,10),'k')
title('迁移-集成测试集误差曲线')
legend('预训练模型','迁移-集成模型')