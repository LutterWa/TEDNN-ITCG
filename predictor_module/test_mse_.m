for i = 0:4
test_mse(i);
end

function [error]=test_mse(p)
load(['dnn_test',num2str(p)]);
error = y-y_hat;
r2 = 1-(sum(error.^2)/sum((y-mean(y)).^2));
%     fprintf(['均值=%f ', '标准差=%f ', '最大值=%f\n'],mean(error), std(error), max(abs(error)))
fprintf(['RMSE=%f ', 'MAE=%f ', 'R2=%f\n'],sqrt(mse(error)), mae(error), r2)
end