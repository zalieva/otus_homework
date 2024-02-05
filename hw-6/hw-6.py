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


# подгружаем данные
data = pd.read_csv('dataset/AB_NYC_2019.csv')

# - Выкиньте ненужные признаки: id, name, host_id, host_name, last_review
data = data.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'])

# Часть 1. EDA
# - Визуализируйте базовые статистики данных: распределения признаков, матрицу попарных корреляций, постройте pair plots

# Визуализация распределений признаков
plt.figure(figsize=(8, 4))
sns.histplot(data['price'], kde=True, color='skyblue')
plt.title('Distribution of Prices')
plt.xlabel('price')
plt.ylabel('Frequency')
# plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data['number_of_reviews'], kde=True, color='salmon')
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
# plt.show()

# Матрица попарных корреляций
correlation_matrix = data[['price', 'number_of_reviews', 'reviews_per_month']].corr()
plt.figure(figsize=(8, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pairwise Correlation Matrix')
# plt.show()

# Pair plots
sns.pairplot(data[['price', 'number_of_reviews', 'reviews_per_month']])
# plt.show()

# - по результатам анализа произведите предобработку переменных

number_of_reviews_bins = data.number_of_reviews.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
data['number_of_reviews_bins'] = pd.cut(data['number_of_reviews'], number_of_reviews_bins, duplicates='drop',
                                   labels=['0-0.3', '0.3-0.4', '0.4-0.5',
                                           '0.5-0.6', '0.6-0.7', '0.7-0.8',
                                           '0.8-0.9', '0.9-1'], right=True, include_lowest=True)

data.reviews_per_month = data.groupby(['number_of_reviews_bins']).reviews_per_month.\
    transform(lambda x: x.fillna(x.mean())).round(2)
print(data.reviews_per_month)


# Часть 2. Preprocessing & Feature Engineering

# работа с категориальными переменными (можно начать с dummy);
# from category_encoders import OrdinalEncoder, OneHotEncoder
# # enc = OneHotEncoder()
# # print(enc.fit_transform(data[['room_type']]).head())
# # data_enc = data.drop(['room_type'], axis=1).join(enc.fit_transform(data[['room_type']], axis=0))
# # print(data_enc.head().columns)

data = pd.get_dummies(data, columns=['room_type'])
print(data.head().values)
print(data.columns)

# замена аномалий;