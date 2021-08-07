import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
from data_preprocessing import data_preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def main(x_train, y_train, x_test, y_test, algorithm='FNN', cross_validation=1, plot_hist=0):
    """[Main function for training]

    Args:
        algorithm (str): [Choose an algorithm]. Defaults to 'FNN'.
        cross_validation (bool): [Whether to do cross-validation]. Defaults to 1.
        plot_hist (bool): [Whether to plot histogram]. Defaults to 0.

    Returns:
        [array]: [Return y_predict_on_train in cross-validation stage and y_predict_on_test in testing stage]
    """
    # (1)数据标准化
    # 特征值标准化
    scale1 = StandardScaler()
    scale1.fit(x_train) #计算并记录x_train的均值和标准差，用其对x_test做相同标准化
    x_train = scale1.transform(x_train) #标准化
    x_test = scale1.transform(x_test)
    # 标签值标准化
    scale2 = StandardScaler()
    scale2.fit(np.array(y_train).reshape(-1,1)) # 当前为行向量，此方法只能对列向量计算
    y_train = scale2.transform(np.array(y_train).reshape(-1,1))
    y_test = scale2.transform(np.array(y_test).reshape(-1,1))
    y_train, y_test = y_train.flatten(), y_test.flatten() # 将列向量转为行向量
    print('---------{}------------------'.format(algorithm))
    # (2)模型训练
    if cross_validation:
        print('---------交叉验证阶段---------')
        # (3)算法选择，设定超参数
        regressor = algorithm_selection(algorithm)
        # (4)交叉验证，调整超参数
        train_sizes, train_loss, valid_loss = learning_curve(regressor, x_train, y_train, cv = 5, scoring='neg_mean_absolute_error',n_jobs=-1)
        # (5)最终模型的训练
        regressor.fit(x_train, y_train) #训练
        y_predict_on_train = regressor.predict(x_train) #预测
        y_train_inverse = scale2.inverse_transform(y_train) #真实值的标准化还原
        y_predict_on_train = scale2.inverse_transform(y_predict_on_train) #预测值的标准化还原
        print('训练集平均绝对误差为：{:.2f}'.format(mean_absolute_error(y_train_inverse,y_predict_on_train)))
        joblib.dump(regressor,f'.\model\{algorithm}.pkl')
        learning_curve_plot(train_sizes, train_loss, valid_loss, algorithm) #绘制学习曲线
        return y_predict_on_train
    else:
    # (6)模型预测与评估（测试误差）
        print('---------测试阶段---------')
        regressor = joblib.load(f'.\model\{algorithm}.pkl')
        y_predict_on_test = regressor.predict(x_test)
        y_test_inverse = scale2.inverse_transform(y_test)
        y_predict_on_test = scale2.inverse_transform(y_predict_on_test)
        print('MAE:  {:.2f}'.format(mean_absolute_error(y_test_inverse,y_predict_on_test)))
        print('MSE:  {:.2f}'.format(mean_squared_error(y_test_inverse,y_predict_on_test)))
        print('RMSE: {:.2f}'.format(mean_squared_error(y_test_inverse,y_predict_on_test) ** 0.5))
        # print('测试集样本上的真实值为：',np.around(y_test_inverse[:10]))
        # print('测试集样本上的预测值为：',np.around(y_predict_on_test[:10],0))
        if plot_hist:
            hist_plot(y_test_inverse, y_predict_on_test, algorithm)
        return y_predict_on_test

def algorithm_selection(algorithm):
    if algorithm == 'FNN':
        from sklearn.neural_network import MLPRegressor
        regressor = MLPRegressor(hidden_layer_sizes=(20,20),max_iter=800,alpha=0.2) #成模型对象
    elif algorithm == 'catboost':
        import catboost as cat
        regressor = cat.CatBoostRegressor(iterations=1000,learning_rate=0.01,depth=10,silent=True)    
    elif algorithm == 'lightgbm':
        import lightgbm as lgb
        regressor = lgb.LGBMRegressor(n_estimators=800,learning_rate=0.01,max_depth=8,reg_alpha=10)
    elif algorithm == 'xgboost':
        import xgboost as xgb
        regressor = xgb.XGBRegressor(max_depth=10, n_estimators=800, learning_rate=0.01, reg_lambda=20)
    elif algorithm == 'RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=1000, max_depth=10)
    elif algorithm == 'SVM':
        from sklearn import svm
        regressor = svm.SVR()
    return regressor

def learning_curve_plot(train_sizes, train_loss, valid_loss, algorithm):
    train_loss_mean = -np.mean(train_loss,1)
    train_loss_std = np.std(train_loss,1)
    valid_loss_mean = -np.mean(valid_loss,1)
    valid_loss_std = np.std(valid_loss,1)
    plt.fill_between(train_sizes, train_loss_mean - train_loss_std,
                    train_loss_mean + train_loss_std, alpha=0.1,color="g")
    plt.fill_between(train_sizes, valid_loss_mean - valid_loss_std,
                     valid_loss_mean + valid_loss_std, alpha=0.1,color="r")
    plt.plot(train_sizes, train_loss_mean, 'o-', c='g', label='Training')
    plt.plot(train_sizes, valid_loss_mean, 'o-', c='r', label='Cross-validation')
    plt.legend(loc='best')
    plt.xlabel('Training examples')
    plt.ylabel('Mean absolute error')
    plt.title(f'Learning Curve of {algorithm}')
    plt.savefig(f'.\picture\Learning Curve of {algorithm}.png')
    plt.show()

def hist_plot(y_test, y_predict_on_test, algorithm):
    plt.figure()
    plt.hist(y_test,10)
    plt.title('True Price Histogram')
    plt.figure()
    plt.hist(y_predict_on_test,10)
    plt.title(f'Predict Price by {algorithm}')
    plt.show()

if __name__ == "__main__":
    start = time.time()
    x_train, y_train, x_test, y_test, *_ = data_preprocessing()
    # 'FNN'、'catboost'、'RandomForest'、'SVM'、'lightgbm'、'xgboost'
    y_predict = main(x_train, y_train, x_test, y_test, algorithm='FNN', cross_validation=0, plot_hist=0)
    print('算法耗时为：{:.2f}s'.format(time.time()-start))