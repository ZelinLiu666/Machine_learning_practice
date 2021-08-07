import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import joblib
import time

def main(algorithm,cross_validation=0):
    # 数据读取和预处理
    data = pd.read_csv('nba_2017_players_with_salary_wiki_twitter.csv')
    data = data[['POSITION', 'FGA', 'FG%', '2PA', '2P%', '3PA', '3P%', 'eFG%', 'FTA', 
                'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS']]
    data = data.dropna(axis=0, how='any')
    # corr_analysis(data) # 绘制相关性分析热力图
    x = data.drop(columns='POSITION')
    y = data.POSITION
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=20)
    # 标准化（特征值进行相同标准化的同时，不向训练集暴露测试集信息）
    scale_features = StandardScaler()
    scale_features.fit(x_train)
    x_train = scale_features.transform(x_train)
    x_test = scale_features.transform(x_test)
    # 训练
    print('---------{}------------------'.format(algorithm))
    if cross_validation:
        print('---------交叉验证阶段---------')
        clf = algorithm_selection(algorithm)
        train_sizes, train_loss, valid_loss = learning_curve(clf, x_train, y_train, cv = 10, scoring='accuracy',n_jobs=-1)
        learning_curve_plot(train_sizes, train_loss, valid_loss, algorithm) #绘制学习曲线
        clf.fit(x_train,y_train)
        joblib.dump(clf,rf'.\model\{algorithm}.pkl')
    else:
        print('---------测试阶段---------')
        clf = joblib.load(rf'.\model\{algorithm}.pkl')
        # 预测
        y_predict_on_test = clf.predict(x_test)
        # 评估
        print('准确率为：{:.2f}%'.format(100 * clf.score(x_test, y_test)))
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_predict_on_test))

def algorithm_selection(algorithm):
    if algorithm == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors = 14)
    if algorithm == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
    if algorithm == 'FNN':    
        from sklearn.neural_network import MLPClassifier 
        clf = MLPClassifier(solver = 'adam',hidden_layer_sizes=(20,), max_iter=200, 
                            alpha=1, learning_rate_init=0.1,verbose=True,early_stopping=True)
    if algorithm == 'SVM': 
        from sklearn import svm
        clf = svm.SVC()
    if algorithm == 'catboost': 
        import catboost as cat
        clf = cat.CatBoostClassifier(iterations=100,learning_rate=0.01,depth=10,silent=True)
    elif algorithm == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
    return clf

def learning_curve_plot(train_sizes, train_loss, valid_loss, algorithm):
    train_loss_mean = np.mean(train_loss,1)
    train_loss_std = np.std(train_loss,1)
    valid_loss_mean = np.mean(valid_loss,1)
    valid_loss_std = np.std(valid_loss,1)
    plt.fill_between(train_sizes, train_loss_mean - train_loss_std,
                    train_loss_mean + train_loss_std, alpha=0.1,color="g")
    plt.fill_between(train_sizes, valid_loss_mean - valid_loss_std,
                     valid_loss_mean + valid_loss_std, alpha=0.1,color="r")
    plt.plot(train_sizes, train_loss_mean, 'o-', c='g', label='Training')
    plt.plot(train_sizes, valid_loss_mean, 'o-', c='r', label='Cross-validation')
    plt.legend(loc='best')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy_score')
    plt.title(f'Learning Curve of {algorithm}')
    plt.savefig(rf'.\picture\Learning Curve of {algorithm}.png')
    plt.show()

def corr_analysis(data):
    corr = data.corr()
    sns.heatmap(corr,cmap='YlGnBu',linewidths=0.5)
    plt.savefig(rf'.\picture\corr.png')
    plt.show()

if __name__ == "__main__":
    start = time.time()
    # 'KNN'、'GaussianNB'、'FNN'、'SVM'、'catboost'、'RandomForest'
    main(algorithm='KNN',cross_validation=0)
    print('算法耗时为：{:.2f}s'.format(time.time()-start))