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