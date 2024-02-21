import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
sns.set_style('darkgrid')


dataset = pd.read_csv('data.csv', index_col=0)
print(dataset.columns)
print(dataset.info())

len_columns = len(dataset.select_dtypes('number').columns)
for i in range(len_columns):
    sns.displot(x=dataset[dataset.select_dtypes('number').columns[i]], hue=dataset.diagnosis, kind="kde")
plt.show()

dataset['diagnosis'] = LabelEncoder().fit_transform(dataset['diagnosis'])
corr = dataset.corr()
plt.figure(figsize=(20, 12))
sns.heatmap(corr, cmap='seismic', annot=True, linewidths=.5, fmt='.2f')
plt.show()

cols = corr[abs(corr['diagnosis']) > 0.75].index.tolist()
print(f'Cильно скоррелированы признаки - {cols}')

sns.pairplot(dataset[cols], diag_kind="kde", hue="diagnosis", palette='viridis')
plt.show()

for i in range(len(dataset.columns)):
    sns.boxplot(y=dataset[dataset.columns[i+1]], hue=dataset.diagnosis)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['diagnosis'], axis=1), dataset['diagnosis'], test_size=0.30, random_state=42, stratify=dataset['diagnosis']
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(y_train.value_counts(normalize=True))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
print(knn)

y_pred_train = knn.predict(X_train_scaled)
y_pred_test = knn.predict(X_test_scaled)

def quality_report(prediction, actual):
    print("Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\nf1_score: {:.3f}".format(
        accuracy_score(prediction, actual),
        precision_score(prediction, actual),
        recall_score(prediction, actual),
        f1_score(prediction, actual)
    ))

print("Train quality:")
quality_report(y_pred_train, y_train)
print("\nTest quality:")
quality_report(y_pred_test, y_test)

roc_auc = roc_auc_score(y_test, y_pred_test)
print(f'ROC-AUC: {roc_auc}')

neighbors = range(1, 50)
f1_score_train = []
f1_score_test = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    f1_score_train.append(f1_score(knn.predict(X_train_scaled), y_train))
    f1_score_test.append(f1_score(knn.predict(X_test_scaled), y_test))

plt.plot(neighbors, f1_score_train, color='blue', label='train')
plt.plot(neighbors, f1_score_test, color='red', label='test')
plt.title("Max test quality: {:.3f}\nBest k: {}".format(max(f1_score_test), np.argmax(f1_score_test) + 1))
plt.legend()
plt.show()


knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train_scaled, y_train)

y_pred_train = knn5.predict(X_train_scaled)
y_pred_test = knn5.predict(X_test_scaled)

print("Train quality:")
quality_report(y_pred_train, y_train)
print("\nTest quality:")
quality_report(y_pred_test, y_test)
