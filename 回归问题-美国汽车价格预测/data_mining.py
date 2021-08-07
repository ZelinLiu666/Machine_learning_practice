import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据读取
df_raw_data = pd.read_csv('USA_cars_datasets.csv')
# 特征选择
df_data = df_raw_data.drop(columns=['Unnamed: 0','vin','lot','country','condition'],axis=1)
# 数据清洗，剔除价格异常的数据
df = df_data[df_data['price'] > 100]

# price & year、mileage
plt.figure(num=1,figsize=(12,8))
plt.subplot(1,2,1)
plt.scatter(df['mileage'],df['price'],lw=1,c='#ff9999')
plt.title('correlation of price & mileage')
plt.xlabel('mileage')
plt.ylabel('price')
plt.subplot(1,2,2)
plt.scatter(df['year'],df['price'],lw=1,c='#9999ff')
plt.title('correlation of price & year')
plt.xlabel('year')
plt.ylabel('price')

# price & title_status
plt.figure(2)
sns.violinplot(x='title_status',y='price',data=df)

# price & Brand、 Model、 Color、 State
plt.figure(3)
df_brand = df.groupby('brand').price.mean().sort_values(ascending = False)
sns.barplot(x='price', y='brand', data=df, capsize=.2,order=df_brand.index)
plt.title('correlation of price & brand')

plt.figure(4)
df_model = df.groupby('model').price.mean().sort_values(ascending = False)
sns.barplot(x='price', y='model', data=df, capsize=.2,order=df_model.index)
plt.title('correlation of price & model')

plt.figure(5)
df_color = df.groupby('color').price.mean().sort_values(ascending = False)
sns.barplot(x='price', y='color', data=df, capsize=.2,order=df_color.index)
plt.title('correlation of price & color')

plt.figure(6)
df_state = df.groupby('state').price.mean().sort_values(ascending = False)
sns.barplot(x='price', y='state', data=df, capsize=.2,order=df_state.index)
plt.title('correlation of price & state')

plt.show()