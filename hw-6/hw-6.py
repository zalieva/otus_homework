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
# Попробуйте на основании имеющихся переменных создать новые, которые могли бы улучшить качество модели. Например, можно найти координаты Манхэттена (самого дорогого района) и при помощи широты и долготы, а также евклидового расстояния создать новую переменную - расстояние от квартиры до этого района. Возможно, такой признак будет работать лучше, чем просто широта и долгота.
# Часть 3. Моделирование

# импортируем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# подгружаем данные
data = pd.read_csv('dataset/AB_NYC_2019.csv')

# - Выкиньте ненужные признаки: id, name, host_id, host_name, last_review
data = data.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'])

# Часть 1. EDA
# - Визуализируйте базовые статистики данных: распределения признаков, матрицу попарных корреляций, постройте pair plots

# # Визуализация распределений признаков

cols = len(data.columns)
data.hist(layout=(cols, 3), figsize=(20, 2.5*cols))
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=data, x='price', y='neighbourhood_group', hue='neighbourhood_group')
plt.show()
plt.figure(figsize=(10,6))
sns.boxplot(data=data, x='price', y='room_type', hue='room_type')
# plt.show()

# plt.figure(figsize=(8, 4))
# sns.histplot(data['price'], kde=True, color='skyblue')
# plt.title('Distribution of Prices')
# plt.xlabel('price')
# plt.ylabel('Frequency')
# plt.show()
#
# plt.figure(figsize=(8, 4))
# sns.histplot(data['number_of_reviews'], kde=True, color='salmon')
# plt.title('Distribution of Number of Reviews')
# plt.xlabel('Number of Reviews')
# plt.ylabel('Frequency')
# # plt.show()
#
# Матрица попарных корреляций


correlation_matrix = data[['price', 'number_of_reviews', 'reviews_per_month']].corr()
plt.figure(figsize=(8, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pairwise Correlation Matrix')
plt.show()

#
# plt.subplots(figsize=(18,15))
# sns.heatmap(data.corr(), cmap=sns.color_palette("coolwarm", 10000), vmin=-1, center=0)
# plt.show()


# cm = np.corrcoef(data[cols].values.T)
# plt.figure(figsize=(10,7))
# sns.set(font_scale=1.25)
# sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12},\
#                 yticklabels=cols, xticklabels=cols, vmin=-1, center=0,\
#                     cmap=sns.color_palette('coolwarm',1000))
# plt.show()

# # Pair plots
# sns.pairplot(data[['price', 'number_of_reviews', 'reviews_per_month']])
# # plt.show()

# - по результатам анализа произведите предобработку переменных
print(data.columns)
# print(np.round(data.isna().sum()[data.isna().sum()>0] / data.shape[0], 2))
#
# print(data.dropna().shape[0] / data.shape[0])
# print(data[['reviews_per_month']].describe())
# print(data[['number_of_reviews']].describe())
#
# number_of_reviews_bins = data.number_of_reviews.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# print(number_of_reviews_bins)
# data['number_of_reviews_bins'] = pd.cut(data['number_of_reviews'], number_of_reviews_bins, duplicates='drop',
#                                    labels=['0-0.3', '0.3-0.4', '0.4-0.5',
#                                            '0.5-0.6', '0.6-0.7','0.7-0.8', '0.8-0.9','0.9-1'], right=True, include_lowest=True)
#
# print(data[['number_of_reviews_bins']].describe())
# plt.figure(figsize=(15,8))

# sns.histplot(data['number_of_reviews_bins'], kde=True, color='skyblue')
# plt.show()
# print(data.reviews_per_month)
# plt.figure(figsize=(15,8))
#
# sns.histplot(data['reviews_per_month'], kde=True, color='skyblue')
# plt.show()
# data.reviews_per_month = data.groupby(['number_of_reviews_bins']).reviews_per_month.\
#     transform(lambda x: x.fillna(x.mean())).round(2)
# data.reviews_per_month = data.reviews_per_month.isna()
data.loc[data['reviews_per_month'].isnull(), 'reviews_per_month'] = 0
# print(data.reviews_per_month)
# plt.figure(figsize=(15,8))
#
# sns.histplot(data['reviews_per_month'], kde=True, color='skyblue')
# plt.show()


# Часть 2. Preprocessing & Feature Engineering

# работа с категориальными переменными (можно начать с dummy);
# from category_encoders import OrdinalEncoder, OneHotEncoder
# # enc = OneHotEncoder()
# # print(enc.fit_transform(data[['room_type']]).head())
# # data_enc = data.drop(['room_type'], axis=1).join(enc.fit_transform(data[['room_type']], axis=0))
# # print(data_enc.head().columns)

data_ = pd.get_dummies(data, columns=['room_type'])
# print(data.head().values)
# print(data_.describe().to_csv('descri'))


# замена аномалий;
# data = data.drop(data.loc[((data.minimum_nights > 365) | (data.price == 0))].index, axis=0)
# print(data)
plt.subplots(figsize=(10,7))
sns.distplot(data['minimum_nights'], label='minimum_nights_in_year')
plt.axvline(data.minimum_nights.quantile(0.95), label='95% quantile', c='mediumslateblue')
plt.axvline(data.minimum_nights.quantile(0.99), label='99% quantile', c='orchid')
plt.legend()
# plt.show()