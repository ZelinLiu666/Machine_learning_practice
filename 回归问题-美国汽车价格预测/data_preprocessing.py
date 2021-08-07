import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def target_encoding(df_data, columns):
    for column in columns:
        df = df_data.groupby(column).price.mean() # 计算各列的所有类别特征值的目标平均值
        for feature_value in df.index:
            df_data.loc[df_data[column]==feature_value,column] = df.loc[feature_value] #将各个类别值替换为目标平均值
    return df_data

def binary_encoding(df_data, columns):
    for column in columns:
        value_0, value_1 = df_data.loc[:,column].unique() # 获取二值特征的类别特征值
        df_data.loc[np.array(df_data.loc[:,column]==value_0), column] = 0
        df_data.loc[np.array(df_data.loc[:,column]==value_1), column] = 1
    return df_data

def data_preprocessing():
    # 数据读取
    df_raw_data = pd.read_csv('USA_cars_datasets.csv')
    # 特征选择
    df_data = df_raw_data.drop(columns=['Unnamed: 0','vin','lot','country','condition'],axis=1)
    # 数据清洗，剔除价格异常的数据
    df = df_data[df_data['price']>100]
    # 数据集划分
    x, y = df.drop(columns='price'), df.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
    # 特征编码
    target_encoding_columns = ['brand','model','color','state']
    binary_columns = ['title_status']
    # 训练集特征编码
    xy_train = pd.concat([x_train,y_train], axis=1)
    xy_train = target_encoding(xy_train, target_encoding_columns)
    xy_train = binary_encoding(xy_train, binary_columns)
    x_train, y_train = xy_train.drop(columns='price'), xy_train.price
    # 测试集特征编码
    xy_test = pd.concat([x_test,y_test], axis=1)
    xy_test = target_encoding(xy_test, target_encoding_columns)
    xy_test = binary_encoding(xy_test, binary_columns)    
    x_test, y_test = xy_test.drop(columns='price'), xy_test.price
    return x_train, y_train, x_test, y_test, df

def correlation_analysis(x_train, y_train, x_test, y_test, flag=True):
    df_train = pd.concat([y_train,x_train],axis=1)
    df_test = pd.concat([y_test,x_test],axis=1)
    df_data = pd.concat([df_train,df_test],axis=0).astype(float)
    corr_pearson = df_data.corr(method='pearson') #分析线性相关关系
    corr_kendall = df_data.corr(method='kendall') #用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
    corr_spearman = df_data.corr(method='spearman') #非线性的，非正太分析的数据的相关系数
    if flag == True:
        plt.figure()
        sns.heatmap(corr_pearson, cmap='YlGnBu',annot=True,linewidths=0.5)
        plt.title('Pearson_Correlation')
        plt.xticks(rotation=45)
    
        plt.figure()
        sns.heatmap(corr_kendall, cmap='YlGnBu',annot=True,linewidths=0.5)
        plt.title('Kendall_Correlation')
        plt.xticks(rotation=45)
        
        plt.figure()
        sns.heatmap(corr_spearman, cmap='YlGnBu',annot=True,linewidths=0.5)
        plt.title('Spearman_Correlation')
        plt.xticks(rotation=45)
        plt.show()
    return df_data

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, df = data_preprocessing()
    # df_data = correlation_analysis(x_train, y_train, x_test, y_test)