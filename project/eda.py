from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import lsimodel, ldamodel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import warnings

import re

warnings.filterwarnings("ignore")
# %matplotlib inline
sns.set_style('darkgrid')

dataset = pd.read_csv('datasets/youtuberesults__RIrYWhjdK_o.csv')
print(dataset.columns)
# print(dataset.head().values)
# print(dataset.likeCount)
# sns.displot(x=dataset.likeCount)
# plt.show()


# dataset['logLikeCount'] = np.log1p(dataset['likeCount'])
# dataset['logLikeCount'].hist(bins=50)
# # plt.show()
# print(dataset.head().values)

# videoPublishedAt = "2022-12-18T16:10:37Z"
# # Приведение даты к типу datetime
# dataset['publishedAt'] = pd.to_datetime(dataset['publishedAt'], format="%Y-%m-%dT%H:%M:%SZ")
# videoPublishedAt = pd.to_datetime(videoPublishedAt, format="%Y-%m-%dT%H:%M:%SZ")
#
# # Разница между датами публикацией комментария и видео
# dataset['publishedDifference'] = (dataset['publishedAt'] - videoPublishedAt).apply(lambda x: x.total_seconds()).astype(int)
#
# print(dataset.head().values)

# # Приведение даты к типу datetime
# dataset['publishedAt'] = pd.to_datetime(dataset['publishedAt'], format="%Y-%m-%dT%H:%M:%SZ")
# dataset['videoPublishedAt'] = pd.to_datetime(dataset['videoPublishedAt'], format="%Y-%m-%dT%H:%M:%SZ")
#
# # Разница между датами публикацией комментария и видео
# dataset['publishedDifference'] = (dataset['publishedAt'] - dataset['videoPublishedAt']).apply(lambda x: x.total_seconds()).astype(int)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

import pymorphy2
import re
import tqdm

nltk.download('stopwords')
nltk.download('punkt')
# nltk.download('omw-1.4')


# Функция обрабатывает тексты для дальнейшего использования
def process_text(texts):
    morph = pymorphy2.MorphAnalyzer()
    def lemmatize(word):
        p = morph.parse(word)[0]
        return p.normal_form

    my_stopword=['это', 'спасибо', 'что', 'очень', 'всё', 'который', 'свой']
    stemmer = SnowballStemmer(language='russian')
    stop_words = stopwords.words('russian')+my_stopword
    print(stop_words)
    # stop_words.add(my_stopword)
    regex = re.compile('[^а-я А-Я]')
    process_texts = []

    for text in tqdm.tqdm(texts):
        if type(text) is not str:
            process_texts.append(text)
            continue
        # Удаляет любые символы, кроме русских букв
        text = regex.sub(' ', text)
        text = text.lower()
        # process_texts.append(text)
        # Разбивает текст на отдельные слова
        word_tokens = word_tokenize(text)
        # Убирает стоп слова и пропускаем через стемминг оставшиеся
        # filtered_sentence = [stemmer.stem(w) for w in word_tokens if not w in stop_words]
        filtered_sentence = []
        for word in word_tokens:
            l_word = lemmatize(word)
            if l_word not in stop_words:
                filtered_sentence.append(l_word)

        # filtered_sentence = [w for w in word_tokens if w not in stop_words]
        process_texts.append(' '.join(filtered_sentence))

    return process_texts



dataset['textProcessed'] = process_text(dataset['textOriginal'])
print(dataset.head().values)

from gensim import corpora
dataset['words'] = dataset['textProcessed'].str.split()
dictionary = corpora.Dictionary(dataset['words'])
print(dictionary)
dictionary.filter_extremes(no_below=10, no_above=0.99, keep_n=None)

corpus = [dictionary.doc2bow(text) for text in dataset['words']]
lsi = lsimodel.LsiModel(corpus, id2word=dictionary, num_topics=7)
pprint(lsi.show_topics(num_words=5, formatted=False))
# print(dataset['words'])

lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=7, update_every=0)
pprint(lda.show_topics(num_words=5, formatted=False))