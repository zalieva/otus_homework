# Пройдите по основным шагам работы с данными:
# выкиньте ненужные признаки: id, name, host_id, host_name, last_review
# визуализируйте базовые статистики данных: распределения признаков, матрицу попарных корреляций, постройте pair plots
# по результатам анализа произведите предобработку переменных
# Часть 2. Preprocessing & Feature Engineering
# Ваша цель получить как можно более высокие метрики качества (можно взять несколько, R2, MAE, RMSE),
# сконцентрировавшись на преобразовании признаков.
# Опробуйте различные техники:
# работа с категориальными переменными (можно начать с dummy);
# замена аномалий;
# различные варианты шкалирования непрерывных переменных (StandardScaler, RobustScaler, и.т.д.);
# обратите внимание на распределение целевой переменной, возможно, с ней тоже можно поработать;
# Попробуйте на основании имеющихся переменных создать новые, которые могли бы улучшить качество модели.
# Например, можно найти координаты Манхэттена (самого дорогого района) и при помощи широты и долготы,
# а также евклидового расстояния создать новую переменную - расстояние от квартиры до этого района.
# Возможно, такой признак будет работать лучше, чем просто широта и долгота.
# Часть 3. Моделирование


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import warnings

warnings.filterwarnings("ignore")
# %matplotlib inline
sns.set_style('darkgrid')

dataset = pd.read_csv('dataset/AB_NYC_2019.csv')

# - Выкиньте ненужные признаки: id, name, host_id, host_name, last_review
dataset = dataset.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review', 'neighbourhood'])
# print(dataset.info())
# Часть 1. EDA
# - Визуализируйте базовые статистики данных: распределения признаков, матрицу попарных корреляций, постройте pair plots

# len_columns = len(dataset.select_dtypes('number').columns)
# for i in range(len_columns):
#     sns.displot(x=dataset[dataset.select_dtypes('number').columns[i]], kind="kde")
# # plt.show()
#
# corr = dataset.select_dtypes('number').corr()пуцвцай
# corr = dataset.select_dtypes('number').corr()пуцвцай
# plt.figure(figsize=(20, 12))
# sns.heatmap(corr, cmap='seismic', annot=True, linewidths=.5, fmt='.2f')
# # plt.show()


# Pair plots
# sns.pairplot(dataset.select_dtypes('number'))
# plt.show()


from sklearn.datasets import fetch_california_housing
import pandas as pd
bunch = fetch_california_housing()
df = pd.DataFrame(bunch['dataset'], columns=bunch['feature_names'])
df['price'] = bunch['price']
print(df.describe())
#
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import StandardScaler
#
# def rmse(y_hat, y):
#     return np.sqrt(mean_squared_error(y_hat, y))
#
# pipe = Pipeline([('scaler', StandardScaler()), ('dt', DecisionTreeRegressor())])
# from sklearn.model_selection import train_test_split, cross_val_score
# X = df.drop('target', axis=1)
# Y = df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
# pipe.fit(X_train, y_train)
# preds = pipe.predict(X_test)
# print('R2: ', r2_score(y_test, preds))
# print('RSME: ', rmse(y_test, preds))


# print(np.round(dataset.isna().sum()[dataset.isna().sum()>0] / dataset.shape[0], 2))
# print(dataset.dropna().shape[0] / dataset.shape[0])
# sns.boxplot(x=dataset['reviews_per_month'].isna(), hue=dataset.number_of_reviews)
# plt.show()
# dataset['reviews_per_month'] = dataset['reviews_per_month'].fillna(0)

# print(dataset.dropna().shape[0] / dataset.shape[0])

# Часть 2. Preprocessing & Feature Engineering
# работа с категориальными переменными (можно начать с dummy);
# from category_encoders import OrdinalEncoder, OneHotEncoder
# # enc = OneHotEncoder()
# # print(enc.fit_transform(data[['room_type']]).head())
# # data_enc = data.drop(['room_type'], axis=1).join(enc.fit_transform(data[['room_type']], axis=0))
# # print(data_enc.head().columns)

# def encode_func(data, enc, cols=['room_type', 'neighbourhood_group']):
#     data_enc = data.copy()
#     data_enc[cols] = enc.fit_transform(data_enc[cols])
#     return data_enc
#

# enc = OrdinalEncoder()
# dataset = encode_func(dataset, enc)
# print(dataset.room_type.unique())
# print(dataset.neighbourhood_group.unique())
# print(dataset.columns)


# # замена аномалий;
# dataset = dataset.drop(dataset.loc[((dataset.minimum_nights > 365) | (dataset.price == 0))].index, axis=0)
# len_columns = len(dataset.select_dtypes('number').columns)
#
#
# for i in range(len_columns):
#     plt.subplots(figsize=(10,7))
#     sns.distplot(x=dataset[dataset.select_dtypes('number').columns[i]], label=dataset.select_dtypes('number').columns[i])
#     plt.axvline(dataset[dataset.select_dtypes('number').columns[i]].quantile(0.95), label='95% quantile', c='mediumslateblue')
#     plt.axvline(dataset[dataset.select_dtypes('number').columns[i]].quantile(0.99), label='99% quantile', c='orchid')
# plt.legend()
# # plt.show()
#
# dataset = dataset.loc[((dataset.reviews_per_month < dataset.reviews_per_month.quantile(0.99)) &
#                        (dataset.price < dataset.price.quantile(0.99)) &
#                        (dataset.minimum_nights < dataset.minimum_nights.quantile(0.99)) &
#                        (dataset.number_of_reviews < dataset.number_of_reviews.quantile(0.99)) &
#                        (dataset.calculated_host_listings_count < dataset.calculated_host_listings_count.quantile(0.99)))]
#
# # различные варианты шкалирования непрерывных переменных (StandardScaler, RobustScaler, и.т.д.)
